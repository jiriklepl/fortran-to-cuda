"""Prepare the sources of an application whose selected loop nests stay on the device.

A ``ResidentProject`` names the application's Fortran sources and the
procedures whose loop nests become compiled entries. ``prepare`` preprocesses
the sources, exports and compiles the entries, puts their calls back in place
and expands the residency directives. What it returns says which file the
resident build compiles for each original source and which entry directories
to add; ``compiler.resident.build`` turns that into executables.
"""
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path

from compiler.resident.configure import configure_sources
from compiler.resident.directives import expand_directives
from compiler.resident.inlining import inline_calls
from compiler.resident.reductions import lower_reductions
from compiler.resident.rewrites import inline_tile_scratch_nests, promote_line_snapshots
from compiler.resident.specialize import integer_parameters, specialize
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
    as it is, for code the activation condition makes unreachable. ``assume``
    lists comparisons ``<expression> == <integer constant>`` that usually hold
    when the procedures run: each procedure gets a second version of its body
    with the branches they decide folded away, used while they hold, so that
    loop nests the branches kept apart can share an entry and a launch (see
    ``compiler.resident.specialize``). The procedures are then exact names.
    ``inline`` names procedures (``module::name``) whose calls from the
    selection's procedures are replaced by their statements, so that the nests
    of several calls in a row can share entries and launches (see
    ``compiler.resident.inlining``); a call that cannot be replaced exactly
    stays a call.
    """

    label: str
    procedures: tuple[str, ...]
    guard: str | None = None
    value_scalars: tuple[str, ...] = ()
    minimum_nests: int = 0
    native_rejections: bool = True
    assume: tuple[str, ...] = ()
    inline: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResidentProject:
    """An application and what of it stays on the device.

    ``sources`` are all Fortran sources that the selected procedures and the
    modules they use live in. ``rewrites`` maps a source file name to the
    rewrites applied to it before export: the names of the compiler's loop
    rewrites (``tile_scratch``, ``line_snapshot``) or functions from text to
    text. ``edit(texts)`` gets the text of every source, by original path,
    after the entry calls are in place and before directives are expanded; it
    returns the texts it changed, and may add new sources under a plain file
    name. It is how directives and residency code reach sources the
    application does not own. ``native(region, original)`` may return the
    replacement for a rejected nest instead of the default, which publishes
    the device values first. ``held`` maps ``(procedure, array name)`` to a
    handle expression for arrays the domain binds itself. ``addressable`` lists
    the sources whose arrays get the TARGET attribute a binding needs, as
    paths or, where a file name is unique, as names; None means every source.
    ``known_host_accesses`` holds the ids of audit findings (see ``audit``)
    the application has accepted; with ``strict_audit`` any other finding is
    an error instead of a line in the report.
    """

    sources: tuple[Path, ...]
    selections: tuple[Selection, ...]
    defines: tuple[str, ...] = ()
    domain: ResidentDomain = DEFAULT_DOMAIN
    rewrites: Mapping[str, tuple[str | Callable[[str], str], ...]] = field(default_factory=dict)
    edit: Callable[[dict], dict] | None = None
    native: Callable[[dict, str], str | None] | None = None
    held: Mapping[tuple[str, str], str] = field(default_factory=dict)
    addressable: tuple[str | Path, ...] | None = None
    known_host_accesses: frozenset[str] = frozenset()
    strict_audit: bool = False




def _rewrite(name, text):
    if callable(name):
        return name(text)
    if name == "tile_scratch":
        text, changes = inline_tile_scratch_nests(text)
    elif name == "line_snapshot":
        text, changes = promote_line_snapshots(text)
    else:
        raise ValueError("Unknown source rewrite: " + name)
    if not changes:
        raise ValueError(f"The {name} rewrite found nothing to change")
    return text


def audit(extracted, packages, selected, known=frozenset()):
    """Host statements of the selected procedures that touch an array an entry binds.

    Compiled entries keep such an array's newest values on the device, so a
    host statement in the same procedure that names it may see stale values
    unless the host copy was brought up to date first. A statement is not
    reported when a residency directive directly precedes it or sits inside
    it, or when all it does with arrays is pass them whole to procedures that
    are selected themselves, directly or through wrappers that only pass them
    on. Each finding carries ``known``: whether its id is in the accepted set.

    The audit looks inside the selected procedures only. It does not follow an
    array into a host procedure that receives it as an argument, and it cannot
    see directives that an edit adds later; both show up as findings to accept.
    """
    bound = {resource["resource"] for package in packages for item in package["entries"]
             for resource in item["region"]["resources"] if resource["rank"]}
    forwarders = extracted.get("forwarders", {})

    def covered(call, visiting=()):
        """A selected procedure, or a wrapper whose arrays all end up in selected ones."""
        if any(fnmatchcase(call, pattern) for pattern in selected):
            return True
        wrapper = forwarders.get(call)
        return bool(wrapper and wrapper["only_calls"] and call not in visiting
                    and all(covered(item, (*visiting, call)) for item in wrapper["calls"]))

    findings = []
    for statement in extracted["host_statements"]:
        arrays = [item["name"] for item in statement["arrays"] if item["resource"] in bound]
        if not arrays or statement["acknowledged"]:
            continue
        if statement["only_calls"] and statement["calls"] and all(covered(call) for call in statement["calls"]):
            continue
        findings.append({"id": statement["id"], "procedure": statement["procedure"],
                         # Lines of the preprocessed source the entries were exported from.
                         "source": statement["source"],
                         "first_line": statement["first_line"], "last_line": statement["last_line"],
                         "arrays": arrays, "calls": statement["calls"], "text": statement["text"],
                         "known": statement["id"] in known})
    return findings


def prepare(project, directory, *, fortran, host_threads=4):
    """Write the resident variant of the application's sources into ``directory``.

    Returns, and writes as ``prepared.json``: ``sources``, one record per
    project source with the ``original`` file and the ``resident`` file the
    resident build compiles in its place (the original itself when nothing
    changed), followed by the sources an edit added, whose ``original`` is
    None; ``entries``, the directories of the compiled entries; and one
    summary per selection; ``audit``, the host statements that touch bound
    arrays (see ``audit``); ``inlined``, the calls a selection asked to replace
    by the callee's statements, each ``inlined`` or with the ``reason`` it
    stays a call; and ``reductions``, the loop nests that only
    accumulate scalars, each with ``compiled`` or the ``reason`` it stays on
    the host (see ``compiler.resident.reductions``). The directory must not
    exist.
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
                text = _rewrite(name, text)
            path = directory / "rewritten" / path.name
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        current[original] = path

    assumed = [selection for selection in project.selections if selection.assume]
    inlining = [selection for selection in project.selections if selection.inline]
    specialized, inlined = set(), []
    if assumed or inlining:
        texts = {original: path.read_text() for original, path in current.items()}
        constants = integer_parameters(texts.values())
    # Calls are replaced first, so that assumed conditions also fold what the calls brought in.
    for selection in inlining:
        for original in originals:
            text, records = inline_calls(texts[original], selection.procedures, selection.inline, constants)
            inlined += records
            if text != texts[original]:
                texts[original] = text
                path = directory / "rewritten" / current[original].name
                path.parent.mkdir(exist_ok=True)
                path.write_text(text)
                current[original] = path
                specialized.add(original)
    if assumed:
        for selection in assumed:
            for name in selection.procedures:
                routine = name.rsplit("::", 1)[-1]
                if any(character in name for character in "*?["):
                    raise ValueError("Assumed conditions need exact procedure names: " + name)
                # The declaration, not a reference: a procedure statement or its END.
                holders = [original for original, text in texts.items()
                           if re.search(rf"(?im)^\s*(?:\w+\s+)*(?:subroutine|function)\s+{re.escape(routine)}\b", text)]
                if len(holders) != 1:
                    raise ValueError(f"Expected one source that defines {name}, found {len(holders)}")
                texts[holders[0]] = specialize(texts[holders[0]], routine, selection.assume, constants)
                path = directory / "rewritten" / current[holders[0]].name
                path.parent.mkdir(exist_ok=True)
                path.write_text(texts[holders[0]])
                current[holders[0]] = path
                specialized.add(holders[0])

    # Nests that only accumulate scalars get a form whose partial results the device can store.
    reductions = []
    for original in originals:
        text = current[original].read_text()
        lowered, records = lower_reductions(text, patterns)
        if lowered != text:
            path = directory / "rewritten" / current[original].name
            path.parent.mkdir(exist_ok=True)
            path.write_text(lowered)
            current[original] = path
            specialized.add(original)
        reductions += [{**record, "source": original} for record in records]
    lowered = [record for record in reductions if "tag" in record]

    def reduction(entry):
        """The rewritten nest an exported entry belongs to: its parameters carry the nest's names."""
        return next((record for record in lowered if record["procedure"] == entry["procedure"]
                     and any(parameter["name"].startswith(record["tag"]) for parameter in entry["parameters"])), None)

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
        group = pass_scalars_by_value(
            group, directory / (selection.label + "-values"), resources=selection.value_scalars,
            names=lambda entry: (reduction(entry) or {}).get("value_scalars", ()))
        package = compile_resident_entries(group, directory / (selection.label + "-entries"),
                                           host_threads=host_threads)
        # A rewritten reduction runs on the device only when all three of its nests compiled. Otherwise
        # its original nest runs, and what the rewrite added is left out of the entries and the report.
        for record in lowered:
            if not any(fnmatchcase(record["procedure"], pattern) for pattern in chosen):
                continue
            own = [item for item in package["entries"] if reduction(item["region"]) is record]
            record["compiled"] = sum(len(item["region"]["source_loop_spans"]) for item in own) == 3
            if record["compiled"]:
                continue
            lines = current[record["source"]].read_text().splitlines()
            mine = [item for item in package["rejected"] if item["procedure"] == record["procedure"]
                    and record["tag"] in "\n".join(lines[item["first_line"] - 1:item["last_line"]])]
            record["reason"] = "; ".join(sorted({item["reason"] for item in mine})) or "its nests were not exported"
            package = {**package, "entries": [item for item in package["entries"] if item not in own],
                       "rejected": [item for item in package["rejected"] if item not in mine]}
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

    findings = audit(extracted, packages, patterns, project.known_host_accesses)
    unknown = [item for item in findings if not item["known"]]
    if unknown and project.strict_audit:
        raise ValueError("Host statements touch arrays that entries keep on the device, with no residency "
                         "directive before them:\n" + "\n".join(
                             f"  {item['id']} {item['procedure']} line {item['first_line']}: {item['text']}"
                             for item in unknown))
    spliced = splice_resident_entries(packages, directory / "spliced", domain=project.domain, native=project.native,
                                      held=project.held)
    replaced = {item["original"]["path"]: Path(item["replacement"]["path"]) for item in spliced["sources"]}
    texts, touched = {}, set()
    for original in originals:
        before = str(current[original].resolve())
        texts[original] = replaced.get(before, current[original]).read_text()
        if before in replaced or project.rewrites.get(original.name) or original in specialized:
            touched.add(original)
    for record in lowered:
        if not record.get("compiled"):
            # Its original nest runs instead, after the device values are published.
            switch = f"logical, parameter :: {record['tag']}compiled = .true."
            if texts[record["source"]].count(switch) != 1:
                raise ValueError("Lost the switch of the rewritten reduction in " + record["procedure"])
            texts[record["source"]] = texts[record["source"]].replace(switch, switch.replace(".true.", ".false."))
    added = {}
    if project.edit is not None:
        for key, text in project.edit(dict(texts)).items():
            if key in texts:
                if text != texts[key]:
                    texts[key] = text
                    touched.add(key)
            elif Path(key).name != str(key):
                raise ValueError("An edit may add sources only under a plain file name: " + str(key))
            else:
                added[str(key)] = text
    bindable_sources = None if project.addressable is None else {
        item if isinstance(item, str) else Path(item).resolve() for item in project.addressable}
    output = directory / "sources"
    output.mkdir()
    records, directives, names = [], 0, set(added)
    for original in originals:
        text, expanded = expand_directives(texts[original], project.domain, origin=original.stem.lower())
        directives += len(expanded)
        changed = original in touched or bool(expanded)
        if bindable_sources is None or original in bindable_sources or original.name in bindable_sources:
            bindable = addressable_arrays(text)
            changed, text = changed or bindable != text, bindable
        resident = original
        if changed:
            name = original.name
            if name in names:
                name = sha256(str(original).encode()).hexdigest()[:12] + "_" + name
            names.add(name)
            resident = output / name
            resident.write_text(text)
        records.append({"original": str(original), "resident": str(resident), "changed": changed})
    for name, text in added.items():
        text, expanded = expand_directives(text, project.domain, origin=Path(name).stem.lower())
        directives += len(expanded)
        (output / name).write_text(text)
        records.append({"original": None, "resident": str(output / name), "changed": True})

    result = {"version": 1, "kind": "resident_application", "defines": list(project.defines),
              "domain": {"module": project.domain.module, "object": project.domain.object,
                         "default": project.domain == DEFAULT_DOMAIN},
              "sources": records, "entries": [item["directory"] for package in packages for item in package["entries"]],
              "selections": summaries, "directives": directives, "audit": findings, "inlined": inlined,
              "reductions": [{key: str(value) if key == "source" else value for key, value in record.items()
                              if key not in {"tag", "value_scalars"}} for record in reductions]}
    (directory / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
