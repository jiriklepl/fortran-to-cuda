"""Adapt an application to the resident build from a description of it in TOML.

The description names the source trees an application is built from, the
sources the compiler reads, the procedures whose loop nests become entries,
the edits that connect the entries to the application's own code, and the
build. ``prepare`` writes the resident sources; ``build`` compiles them with
the application's native control. ``python -m compiler.resident`` runs both.

A description (paths are relative to the file; ``input:path`` names a file
of an input tree)::

    name = "ELMM"                          # the CMake project

    [inputs.elmm]                          # a source tree, given as --input elmm=PATH
    url = "https://..."                    # where its pinned revision comes from
    revision = "ddb426f..."                # a git checkout of it, or of a descendant, is required
    copy = ["src"]                         # what of it the build snapshots; without, it is used in place
    suffixes = [".f90", ".c"]              # only files with these suffixes are copied
    clean = true                           # a git checkout of exactly ``revision``, without changes

    [variants.dp]                          # --variant dp; the first is the default
    defines = ["DPREC"]

    sources = ["elmm:src/a.f90", ...]      # what the compiler reads, in the build's order
    program = "elmm:src/main.f90"          # the one of them that is the main program: in no library
    addressable = ["elmm:src/a.f90", ...]  # sources whose arrays get the TARGET a binding needs

    [domain]                               # the residency domain (see ResidentDomain)
    module = "..."; object = "..."; check = "..."; prefix = "rk"

    [[held]]                               # arrays the domain binds itself
    procedure = "m::p"; array = "a"; handle = "state%a"

    [[selection]]                          # see Selection
    label = "rk"; procedures = [...]; minimum_nests = 131
    assume = [...]; inline = [...]; native_rejections = true

    [rewrites]                             # the compiler's loop rewrites, by file name
    "advection.f90" = ["tile_scratch"]

    [[patch]]    # edits of the snapshot itself: the native and the resident build see them
    [[rewrite]]  # edits of the preprocessed sources, before the compiler's passes
    [[edit]]     # edits after the entry calls are in place; may add residency directives

An edit names a ``file`` (a source's file name) and, optionally, the
``procedure`` it is confined to (``function = true`` for a function). It
locates a place by one of ``at`` (a piece of text, found once, tolerant of
layout), ``between`` (two pieces of text: from the start of the first to the
end of the second), ``call`` (the CALL statement of that procedure) or
``module_end`` (before ``END MODULE`` of that module), or, with none of
them, the whole procedure. ``action`` is ``before``, ``after`` or
``replace``; ``text`` or ``template`` (a file) is what is inserted, where a
replacement's ``{original}`` stands for what it replaces.

The build::

    [build]
    fortran_options = ["-O3", "-fopenmp"]
    include = ["elmm:src"]                 # include directories, besides the runtime's
    extra = ["elmm:src/lapack.f"]          # sources the compiler does not read, in both libraries
    resident = ["sources/adapter.f90"]     # the application's own residency sources, in the resident library
    libraries = ["fftw3", "blas"]          # system libraries both libraries link
    link = ["OpenMP::OpenMP_Fortran"]      # targets both libraries link
    resident_link = ["CUDA::cudart"]       # targets the resident library links besides
    cmake = '''add_subdirectory({input:solver} solver)'''   # CMake lines before the libraries
    [build.source_options]
    "elmm:src/x.f90" = ["-O0"]

    [[executable]]                         # one per program to build
    name = "app"; library = "resident"; edits = ["timing"]
    [[program_edit.timing]]                # edits of the main program, by name
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from compiler.resident.anchors import find_call_span, find_routine_span, find_unique_span
from compiler.resident.project import DEFAULT_DOMAIN, ResidentProject, Selection
from compiler.resident.project import prepare as prepare_project
from compiler.scopes.resident_splice import ResidentDomain

_ACTIONS = {"before", "after", "replace"}
_LOCATORS = ("at", "between", "call", "module_end")


@dataclass(frozen=True)
class Edit:
    """One anchored change of a source (see the module's description)."""

    file: str
    action: str
    text: str
    procedure: str | None = None
    function: bool = False
    at: str | None = None
    between: tuple[str, str] | None = None
    call: str | None = None
    module_end: str | None = None


@dataclass(frozen=True)
class Input:
    """A source tree an application is built from."""

    name: str
    url: str | None = None
    revision: str | None = None
    copy: tuple[str, ...] | None = None
    suffixes: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    clean: bool = False


@dataclass(frozen=True)
class Executable:
    name: str
    library: str
    edits: tuple[str, ...] = ()


@dataclass(frozen=True)
class Build:
    fortran_options: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    resident: tuple[str, ...] = ()
    libraries: tuple[str, ...] = ()
    link: tuple[str, ...] = ()
    resident_link: tuple[str, ...] = ()
    cmake: str = ""
    source_options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Application:
    """An application's description, read from its TOML file (see ``load``)."""

    root: Path
    name: str
    inputs: dict
    variants: dict
    sources: tuple[str, ...]
    program: str
    addressable: tuple[str, ...]
    domain: ResidentDomain
    held: dict
    selections: tuple[Selection, ...]
    rewrites: dict
    patches: tuple[Edit, ...]
    source_rewrites: tuple[Edit, ...]
    edits: tuple[Edit, ...]
    build: Build
    executables: tuple[Executable, ...]
    program_edits: dict

    def inputs_of(self, names):
        """The inputs that the ``input:path`` names refer to."""
        return {name.split(":", 1)[0] for name in names if ":" in name}


def _keys(table, allowed, where):
    unknown = set(table) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {where}: " + ", ".join(sorted(unknown)))


def _edit(table, root, where):
    _keys(table, {"file", "procedure", "function", "action", "text", "template", *_LOCATORS}, where)
    if table.get("action") not in _ACTIONS:
        raise ValueError(f"{where} needs an action: before, after or replace")
    if sum(key in table for key in _LOCATORS) > 1:
        raise ValueError(f"{where} locates its place in more than one way")
    if not any(key in table for key in _LOCATORS) and "procedure" not in table:
        raise ValueError(f"{where} locates no place")
    if ("text" in table) == ("template" in table):
        raise ValueError(f"{where} needs either a text or a template")
    text = table["text"] if "text" in table else (root / table["template"]).read_text()
    between = tuple(table["between"]) if "between" in table else None
    if between is not None and len(between) != 2:
        raise ValueError(f"{where}: between names two pieces of text")
    return Edit(table["file"], table["action"], text, table.get("procedure"), table.get("function", False),
                table.get("at"), between, table.get("call"), table.get("module_end"))


def load(path):
    """Read and check an application's description."""
    path = Path(path).resolve(strict=True)
    root = path.parent
    value = tomllib.loads(path.read_text())
    _keys(value, {"name", "inputs", "variants", "sources", "program", "addressable", "domain", "held", "selection",
                  "rewrites", "patch", "rewrite", "edit", "build", "executable", "program_edit"}, path.name)
    inputs = {}
    for name, table in value.get("inputs", {}).items():
        _keys(table, {"url", "revision", "copy", "suffixes", "names", "clean"}, f"inputs.{name}")
        inputs[name] = Input(name, table.get("url"), table.get("revision"),
                             tuple(table["copy"]) if "copy" in table else None, tuple(table.get("suffixes", ())),
                             tuple(table.get("names", ())), table.get("clean", False))
    variants = {name: tuple(table.get("defines", ())) for name, table in value.get("variants", {}).items()}
    selections = []
    for table in value.get("selection", ()):
        _keys(table, {"label", "procedures", "minimum_nests", "native_rejections", "assume", "inline"}, "selection")
        selections.append(Selection(table["label"], tuple(table["procedures"]), table.get("minimum_nests", 0),
                                    table.get("native_rejections", True), tuple(table.get("assume", ())),
                                    tuple(table.get("inline", ()))))
    domain = value.get("domain")
    if domain is not None:
        _keys(domain, {"module", "object", "check", "prefix"}, "domain")
        domain = ResidentDomain(**domain)
    held = {}
    for table in value.get("held", ()):
        _keys(table, {"procedure", "array", "handle"}, "held")
        held[table["procedure"], table["array"]] = table["handle"]
    build = value.get("build", {})
    _keys(build, {"fortran_options", "include", "extra", "resident", "libraries", "link", "resident_link", "cmake",
                  "source_options"}, "build")
    build = Build(**{key: tuple(item) if isinstance(item, list) else item for key, item in build.items()})
    executables = []
    for table in value.get("executable", ()):
        _keys(table, {"name", "library", "edits"}, "executable")
        if table["library"] not in {"native", "resident"}:
            raise ValueError(f"Executable {table['name']} needs the native or the resident library")
        executables.append(Executable(table["name"], table["library"], tuple(table.get("edits", ()))))
    program_edits = {name: tuple(_edit(table, root, f"program_edit.{name}") for table in tables)
                     for name, tables in value.get("program_edit", {}).items()}
    unknown = {name for executable in executables for name in executable.edits} - set(program_edits)
    if unknown:
        raise ValueError("Executables name program edits that are not described: " + ", ".join(sorted(unknown)))
    application = Application(
        root, value["name"], inputs, variants, tuple(value["sources"]), value["program"],
        tuple(value.get("addressable", ())), domain, held, tuple(selections),
        {name: tuple(items) for name, items in value.get("rewrites", {}).items()},
        tuple(_edit(table, root, "patch") for table in value.get("patch", ())),
        tuple(_edit(table, root, "rewrite") for table in value.get("rewrite", ())),
        tuple(_edit(table, root, "edit") for table in value.get("edit", ())), build, tuple(executables), program_edits)
    if application.program not in application.sources:
        raise ValueError("The program is one of the sources: " + application.program)
    unknown = application.inputs_of((*application.sources, *application.addressable,
                                     *build.include, *build.extra, *build.source_options)) - set(inputs)
    if unknown:
        raise ValueError("Paths name inputs that are not described: " + ", ".join(sorted(unknown)))
    return application


def apply_edit(text, edit):
    """The text with one edit made; raises ValueError where its place is not found exactly once."""
    low, high = 0, len(text)
    if edit.procedure is not None:
        low, high = find_routine_span(text, edit.procedure, kind="function" if edit.function else "subroutine")
    region = text[low:high]
    if edit.at is not None:
        start, end = find_unique_span(region, edit.at, message=f"{edit.file}: expected one {edit.at!r}")
        start, end = low + start, low + end
    elif edit.between is not None:
        start = find_unique_span(region, edit.between[0], message=f"{edit.file}: expected one {edit.between[0]!r}")[0]
        tail = find_unique_span(region[start:], edit.between[1],
                                message=f"{edit.file}: expected one {edit.between[1]!r}")[1]
        start, end = low + start, low + start + tail
    elif edit.call is not None:
        start, end, _raw, _arguments = find_call_span(text, edit.call, within_routine=edit.procedure)
    elif edit.module_end is not None:
        match = re.search(rf"(?im)^\s*end\s*module\s+{re.escape(edit.module_end)}\b", text)
        if match is None:
            raise ValueError(f"{edit.file}: no END MODULE {edit.module_end}")
        start = end = match.start()
    else:
        start, end = low, high
    if edit.action == "before":
        return text[:start] + edit.text + text[start:]
    if edit.action == "after":
        return text[:end] + edit.text + text[end:]
    return text[:start] + edit.text.replace("{original}", text[start:end]) + text[end:]


def _edited(texts, edits, locate):
    """Apply edits to the texts of the paths ``locate`` gives for their files; returns the changed texts."""
    changed = {}
    for edit in edits:
        path = locate(edit.file)
        if path not in texts:
            raise ValueError("An edit names a file that is not a source: " + edit.file)
        changed[path] = apply_edit(changed.get(path, texts[path]), edit)
    return changed


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments], capture_output=True, text=True,
                          check=True).stdout.strip()


def _check_revision(name, source, spec):
    """An input is a git checkout of its revision or of a descendant; a ``clean`` one is exactly that revision."""
    try:
        head = _git(source, "rev-parse", "HEAD")
        based = subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", spec.revision, head],
                               capture_output=True, check=False).returncode == 0
        changed = spec.clean and _git(source, "status", "--porcelain", "--untracked-files=all")
    except subprocess.CalledProcessError:
        raise ValueError(f"Input {name} must be a git checkout based on {spec.revision}: {source}") from None
    if spec.clean and (head != spec.revision or changed):
        raise ValueError(f"Input {name} must be a clean checkout of {spec.revision}: {source}")
    if not based:
        raise ValueError(f"Input {name} must be based on {spec.revision}: {source}")


def snapshot_inputs(application, given, directory, names):
    """Copy the named inputs that have a ``copy`` list into ``directory``; return the path of each input.

    An input without one is used in place. An input with a ``revision`` is checked against it first.
    """
    paths, records = {}, {}
    for name in sorted(names):
        spec = application.inputs[name]
        if name not in given:
            raise ValueError(f"Missing --input {name}=PATH")
        source = Path(given[name]).resolve(strict=True)
        if spec.revision is not None:
            _check_revision(name, source, spec)
        if spec.copy is None:
            paths[name] = source
            continue
        target = directory / name
        files = []
        for part in spec.copy:
            root = source / part
            for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
                if path.is_file() and (not spec.suffixes or path.suffix.lower() in spec.suffixes
                                       or path.name in spec.names):
                    files.append(path)
        records[name] = []
        for path in files:
            output = target / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            before = _identity(path)
            shutil.copyfile(path, output)
            if _identity(path) != before or _identity(output)["sha256"] != before["sha256"]:
                raise ValueError("Source changed during snapshot: " + str(path))
            records[name].append({"original": before, "snapshot": _identity(output)})
        paths[name] = target
    return paths, records


def _resolve(name, paths, root):
    """The path an ``input:path`` name, or a path relative to the description, stands for."""
    if ":" in name:
        source, relative = name.split(":", 1)
        return paths[source] / relative
    return root / name


def prepare(application, given, directory, *, variant, fortran, jobs=1):
    """Snapshot the inputs and write the resident sources and entries into ``directory``; return the record."""
    if variant not in application.variants:
        raise ValueError(f"Unknown variant {variant!r}: expected one of " + ", ".join(application.variants))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    needed = application.inputs_of((*application.sources, *application.addressable))
    paths, records = snapshot_inputs(application, given, directory / "application", needed)
    sources = [_resolve(name, paths, application.root) for name in application.sources]
    program = _resolve(application.program, paths, application.root)

    def locate(name):
        """The source an edit names: ``input:path``, or a file name no other source has."""
        if ":" in name:
            return _resolve(name, paths, application.root)
        found = [path for path in sources if path.name == name]
        if len(found) != 1:
            raise ValueError(f"An edit names {name}, which is not the name of exactly one source")
        return found[0]

    patched = _edited({path: path.read_text() for path in sources}, application.patches, locate)
    patches = []
    for path, text in patched.items():
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Refusing to patch uncopied input file in place: {path}")
        before = _identity(path)
        path.write_text(text)
        patches.append({"before": before, "after": _identity(path)})
    (directory / "application").mkdir(exist_ok=True)
    (directory / "application" / "source-inputs.json").write_text(
        json.dumps({"inputs": records, "patches": patches}, indent=2) + "\n")

    def rewrite(edits):
        def apply(text, edits=edits):
            for edit in edits:
                text = apply_edit(text, edit)
            return text
        return apply

    # The project names the sources it rewrites by file name.
    rewrites = {locate(name).name: tuple(items) for name, items in application.rewrites.items()}
    for edit in application.source_rewrites:
        name = locate(edit.file).name
        rewrites[name] = (*rewrites.get(name, ()), rewrite((edit,)))

    def edit(texts):
        """The application's own residency code, added once the entry calls are in place."""
        return _edited(texts, application.edits, locate)

    project = ResidentProject(
        sources=tuple(sources), selections=application.selections, defines=application.variants[variant],
        domain=application.domain or DEFAULT_DOMAIN, rewrites=rewrites, edit=edit if application.edits else None, held=application.held,
        addressable=tuple(_resolve(name, paths, application.root) for name in application.addressable))
    prepared = prepare_project(project, directory / "resident", fortran=fortran, jobs=jobs)
    for row in prepared["selections"]:
        print("Generated", row["label"], row["entries"], "entries covering", row["nests"], "source nests", flush=True)
    labels = [selection.label for selection in application.selections]
    (directory / "packages.json").write_text(json.dumps(
        [json.loads((directory / "resident" / (label + "-entries") / "device-entries.json").read_text())
         for label in labels], indent=2) + "\n")
    # Where host statements of the selected procedures get arrays back from the device, for reading.
    (directory / "host-accesses.json").write_text(json.dumps(
        [{key: item[key] for key in ("procedure", "arrays", "text", "access")} for item in prepared["audit"]],
        indent=2) + "\n")
    result = {"version": 3, "application": application.name, "variant": variant,
              "inputs": {name: str(path) for name, path in paths.items()},
              "sources": [{**row, "sha256": _identity(row["resident"])["sha256"]} for row in prepared["sources"]],
              "program": str(program), "entry_directories": prepared["entries"],
              "packages": str(directory / "packages.json")}
    (directory / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Prepared", len(result["entry_directories"]), "GPU entries", flush=True)
    return result


def build(application, prepared, given, directory, *, fortran, cxx, nvcc, architecture, jobs, strict_fp=True):
    """Compile the native and resident libraries and the executables of a prepared application."""
    from compiler.resident.build import Executable as CMakeExecutable
    from compiler.resident.build import (
        ObjectLibrary,
        cmake_lists,
        cmake_path,
        configure_and_compile,
        resident_sources,
        snapshot_directory,
    )

    if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", architecture):
        raise ValueError("An explicit CUDA architecture is required")
    directory = Path(directory).resolve()
    for row in prepared["sources"]:
        if _identity(row["resident"])["sha256"] != row["sha256"]:
            raise ValueError("Prepared application source changed: " + row["resident"])
    paths = {name: Path(path) for name, path in prepared["inputs"].items()}
    others = set(application.inputs) - set(paths)
    more, _records = snapshot_inputs(application, given, directory / "application", others)
    paths.update(more)
    runtime = snapshot_directory(Path(__file__).resolve().parents[1] / "runtime", directory / "runtime")
    # The application's own files, as the build compiles them.
    own = snapshot_directory(application.root, directory / "own")
    program = Path(prepared["program"])
    resident_program = next(Path(row["resident"]) for row in prepared["sources"] if row["original"] == str(program))
    extra = [path for path in (_resolve(name, paths, own) for name in application.build.extra) if path.is_file()]
    native = [Path(row["original"]) for row in prepared["sources"]
              if row["original"] is not None and Path(row["original"]) != program] + extra
    compiled = json.loads((directory / "resident" / "prepared.json").read_text())
    resident = resident_sources(compiled, runtime, programs=(program,)) + extra
    resident += [_resolve(name, paths, own) for name in application.build.resident]
    mains = {}
    for executable in application.executables:
        text = (resident_program if executable.library == "resident" else program).read_text()
        for name in executable.edits:
            for edit in application.program_edits[name]:
                text = apply_edit(text, edit)
        mains[executable.name] = directory / (executable.name + ".f90")
        mains[executable.name].write_text(text)
    prologue = []
    for line in application.build.cmake.splitlines():
        for name, path in paths.items():
            line = line.replace("{input:" + name + "}", cmake_path(path))
        if line.strip():
            prologue.append(line)
    libraries = application.build.libraries
    prologue += [f"find_library(LIB_{lib} NAMES {lib} lib{lib}.so.3 lib{lib}.so.0 REQUIRED)" for lib in libraries]
    prologue += ["add_library(native_libraries INTERFACE)",
                 "target_link_libraries(native_libraries INTERFACE "
                 + " ".join([*application.build.link, *("${LIB_" + lib + "}" for lib in libraries)]) + ")"]
    include = [_resolve(name, paths, own) for name in application.build.include]
    resident_by_original = {
        Path(row["original"]).resolve(): Path(row["resident"]).resolve()
        for row in prepared["sources"]
        if row["original"] is not None
    }
    resolved_source_options = {}
    for name, options in application.build.source_options.items():
        resolved = _resolve(name, paths, own).resolve()
        resolved_source_options[resolved] = tuple(options)
        if resolved in resident_by_original:
            resolved_source_options[resident_by_original[resolved]] = tuple(options)
    text = cmake_lists(
        application.name,
        [ObjectLibrary("native", tuple(native), link=("native_libraries",)),
         ObjectLibrary("resident", tuple(resident), link=("native_libraries", *application.build.resident_link))],
        [CMakeExecutable(name, source, application.executables[index].library)
         for index, (name, source) in enumerate(mains.items())],
        include_directories=(*include, runtime),
        definitions=application.variants[prepared["variant"]],
        fortran_options=application.build.fortran_options, prologue=prologue,
        source_options=resolved_source_options,
        strict_fp=strict_fp)
    (directory / "CMakeLists.txt").write_text(text)
    inputs = {str(path): _identity(path)["sha256"] for path in {*native, *resident, *mains.values()}}
    output = configure_and_compile(directory, fortran=fortran, cxx=cxx, nvcc=nvcc, architecture=architecture, jobs=jobs)
    if any(_identity(path)["sha256"] != digest for path, digest in inputs.items()):
        raise ValueError("Build source changed during compilation")
    result = {"status": "PASS", "strict_fp": strict_fp,
              "binaries": {name: _identity(output / name) for name in mains},
              "libraries": {path.name: _identity(path) for path in sorted((output / "lib").glob("*.so"))}
              if (output / "lib").is_dir() else {}}
    (directory / "built.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Built native and resident executables:", output, flush=True)
    return result
