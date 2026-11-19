"""Prepare the sources of an application whose selected loop nests stay on the device.

A ``ResidentProject`` names the application's Fortran sources and the
procedures whose loop nests become compiled entries. ``prepare`` preprocesses
the sources, exports and compiles the entries, puts their calls back in place
and expands the residency directives. What it returns says which file the
resident build compiles for each original source and which entry directories
to add; ``compiler.resident.build`` turns that into executables.
"""
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path

from compiler.resident.configure import configure_sources
from compiler.resident.directives import expand_directives
from compiler.resident.rewrites import inline_tile_scratch_nests, promote_line_snapshots
from compiler.scopes.resident_entries import (
    compile_resident_entries,
    export_resident_entries,
    guard_entries,
    pass_scalars_by_value,
)
from compiler.scopes.resident_splice import ResidentDomain, addressable_arrays, splice_resident_entries

# The domain of an application that keeps no residency state of its own; the
# runtime provides this module (``resident_application.f90``).
DEFAULT_DOMAIN = ResidentDomain("fort_resident_application", "resident", "resident_require")


@dataclass(frozen=True)
class Selection:
    """Procedures whose loop nests become entries, and how their entries are compiled.

    ``procedures`` are ``module::name`` values or globs over them. ``guard`` is
    a condition the application's activation implies and the loop bounds do not
    show (see ``guard_entries``). ``value_scalars`` are module scalars
    (``module::name``) that are defined before any entry runs and so may be
    passed by value. ``minimum_nests`` is the number of source loop nests that
    must end up compiled; fewer is an error, so lost coverage is not silent.
    With ``native_rejections`` off, a nest the compiler rejects is left exactly
    as it is, for code the activation condition makes unreachable.
    """

    label: str
    procedures: tuple[str, ...]
    guard: str | None = None
    value_scalars: tuple[str, ...] = ()
    minimum_nests: int = 0
    native_rejections: bool = True


@dataclass(frozen=True)
class ResidentProject:
    """An application and what of it stays on the device.

    ``sources`` are all Fortran sources that the selected procedures and the
    modules they use live in. ``rewrites`` maps a source file name to the loop
    rewrites applied to it before export (``tile_scratch``, ``line_snapshot``).
    ``edit(name, text)`` may change a source after the entry calls are in
    place and before directives are expanded, so it can add directives to
    sources the application does not own. ``native(region, original)`` may
    return the replacement for a rejected nest instead of the default, which
    publishes the device values first. ``addressable`` names the files whose
    arrays get the TARGET attribute a binding needs; None means every source.
    """

    sources: tuple[Path, ...]
    selections: tuple[Selection, ...]
    defines: tuple[str, ...] = ()
    domain: ResidentDomain = DEFAULT_DOMAIN
    rewrites: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    edit: Callable[[str, str], str] | None = None
    native: Callable[[dict, str], str | None] | None = None
    addressable: tuple[str, ...] | None = None


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _rewrite(name, text, domain):
    if name == "tile_scratch":
        text, changes = inline_tile_scratch_nests(text)
    elif name == "line_snapshot":
        text, changes = promote_line_snapshots(text, domain=domain)
    else:
        raise ValueError("Unknown source rewrite: " + name)
    if not changes:
        raise ValueError(f"The {name} rewrite found nothing to change")
    return text


def prepare(project, directory, *, fortran, host_threads=4):
    """Write the resident variant of the application's sources into ``directory``.

    Returns, and writes as ``prepared.json``: ``sources``, one record per
    project source with the ``original`` file and the ``resident`` file the
    resident build compiles in its place (the original itself when nothing
    changed); ``entries``, the directories of the compiled entries; and one
    summary per selection. The directory must not exist.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    labels = [selection.label for selection in project.selections]
    if len(set(labels)) != len(labels):
        raise ValueError("Selections need distinct labels")
    patterns = [name.lower() for selection in project.selections for name in selection.procedures]
    configured = configure_sources(project.sources, patterns, directory / "configured", fortran=fortran,
                                   defines=project.defines)
    originals = [Path(item["original"]["path"]) for item in configured["sources"]]
    unknown = set(project.rewrites) - {path.name for path in originals}
    if unknown:
        raise ValueError("Rewrites name files that are not project sources: " + ", ".join(sorted(unknown)))
    current = {}
    for item, original in zip(configured["sources"], originals, strict=True):
        path = Path(item["prepared"]["path"])
        passes = project.rewrites.get(original.name, ())
        if passes:
            text = path.read_text()
            for name in passes:
                text = _rewrite(name, text, project.domain)
            path = directory / "rewritten" / path.name
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        current[original] = path

    extracted = export_resident_entries(list(current.values()), patterns, directory / "extracted")
    packages, summaries = [], []
    for selection in project.selections:
        chosen = [name.lower() for name in selection.procedures]

        def selected(item, chosen=chosen):
            return any(fnmatchcase(item["procedure"], pattern) for pattern in chosen)

        group = {**extracted, "entries": [item for item in extracted["entries"] if selected(item)],
                 "rejected": [item for item in extracted["rejected"] if selected(item)]}
        if selection.guard is not None:
            group = guard_entries(group, directory / (selection.label + "-guarded"), selection.guard)
        group = pass_scalars_by_value(group, directory / (selection.label + "-values"),
                                      resources=selection.value_scalars)
        package = compile_resident_entries(group, directory / (selection.label + "-entries"),
                                           host_threads=host_threads)
        nests = sum(len(item["region"]["source_loop_spans"]) for item in package["entries"])
        if nests < selection.minimum_nests:
            raise ValueError(f"Selection {selection.label} compiled {nests} loop nests; "
                             f"at least {selection.minimum_nests} are required")
        summary = {"label": selection.label, "entries": len(package["entries"]), "nests": nests,
                   "rejected": package["rejected"]}
        if not selection.native_rejections:
            package = {**package, "rejected": []}
        packages.append(package)
        summaries.append(summary)

    spliced = splice_resident_entries(packages, directory / "spliced", domain=project.domain, native=project.native)
    replaced = {item["original"]["path"]: Path(item["replacement"]["path"]) for item in spliced["sources"]}
    output = directory / "sources"
    output.mkdir()
    records, directives, names = [], 0, set()
    for original in originals:
        before = current[original]
        text = replaced.get(str(before.resolve()), before).read_text()
        changed = str(before.resolve()) in replaced or bool(project.rewrites.get(original.name))
        if project.edit is not None:
            edited = project.edit(original.name, text)
            changed, text = changed or edited != text, edited
        text, expanded = expand_directives(text, project.domain, origin=original.stem.lower())
        directives += len(expanded)
        changed = changed or bool(expanded)
        if project.addressable is None or original.name in project.addressable:
            bindable = addressable_arrays(text)
            changed, text = changed or bindable != text, bindable
        resident = original
        if changed:
            name = original.name if original.name not in names else sha256(str(original).encode()).hexdigest()[:12] \
                + "_" + original.name
            names.add(name)
            resident = output / name
            resident.write_text(text)
        records.append({"original": str(original), "resident": str(resident), "changed": changed})

    result = {"version": 1, "kind": "resident_application", "defines": list(project.defines),
              "domain": {"module": project.domain.module, "object": project.domain.object,
                         "default": project.domain == DEFAULT_DOMAIN},
              "sources": records, "entries": [item["directory"] for package in packages for item in package["entries"]],
              "selections": summaries, "directives": directives}
    (directory / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
