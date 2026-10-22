"""Compose independently checked roots against the same original source tree.

Composition merges original-coordinate edit ledgers, never applies a transform
to a previous transform's source. Conflicting helpers require a shared compiler
owner and are rejected; identical artifacts and declarations are shared.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath

from compiler.ir import CompilationError


def _digest(text):
    return sha256(text.encode()).hexdigest()


def _artifact_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise CompilationError("scope package artifact escapes its output directory")
    return path


def _apply(source, edits):
    lines = source.splitlines(keepends=True)
    grouped = {}
    for edit in edits:
        first, last = edit["first_line"], edit["last_line"]
        if not (
            type(first) is int
            and type(last) is int
            and 1 <= first <= len(lines) + 1
            and first - 1 <= last <= len(lines)
        ):
            raise CompilationError("scope composition contains an invalid original source span")
        values = grouped.setdefault((first, last), [])
        text = edit["replacement"]
        if text not in values:
            values.append(text)
    previous = len(lines) + 1
    replacements = sorted((first, last) for first, last in grouped if last >= first)
    for (_first, last), (next_first, _next_last) in zip(replacements, replacements[1:], strict=False):
        if last >= next_first:
            raise CompilationError("scope roots contain overlapping original source replacements")
    for first, last in grouped:
        if last < first and any(begin < first <= end for begin, end in replacements):
            raise CompilationError("scope root insertion is inside another root's source replacement")
    for (first, last), values in sorted(grouped.items(), reverse=True):
        if last >= previous or (last >= first and len(values) != 1):
            raise CompilationError("scope roots contain conflicting original source edits; use one shared owner")
        lines[first - 1 : last] = ["".join(values)]
        if last >= first:
            previous = first
    return "".join(lines)


def compose_graph_scopes(components):
    """Return ``(outputs, manifest)`` for checked ``(outputs, manifest)`` roots.

    Every component must use the same runtime and unchanged original source
    bytes. Each original source is compiled once after its edit union is
    checked. Component manifests remain embedded for proof/provenance review.
    """
    components = tuple(components)
    if not components:
        raise CompilationError("scope composition requires at least one checked component")
    outputs, sources, originals, edits, build_sources, inputs = {}, {}, {}, {}, {}, {}
    manifests = []
    runtime = components[0][1].get("runtime")
    if not runtime:
        raise CompilationError("scope composition requires a compiled shared runtime")
    for artifacts, report in components:
        if report.get("schema_version") != 1 or report.get("runtime") != runtime:
            raise CompilationError("scope components require the same checked runtime")
        if any(item["entry"] == report["entry"] for item in manifests):
            raise CompilationError("scope composition contains a duplicate root")
        if set(artifacts) - set(report.get("artifacts_sha256", {})) - {"scope-manifest.json"}:
            raise CompilationError("scope component includes artifacts absent from its compiler manifest")
        for name, digest in report.get("artifacts_sha256", {}).items():
            _artifact_path(name)
            if name not in artifacts or _digest(artifacts[name]) != digest:
                raise CompilationError("scope component artifact does not match its compiler manifest: " + name)
        for name, digest in report["source_inputs"].items():
            if name in inputs and inputs[name] != digest:
                raise CompilationError("scope roots were compiled against different original inputs")
            inputs[name] = digest
        component_edits = {}
        for edit in report["source_edits"]:
            name = edit["source"]
            if edit["source_sha256"] != report["source_inputs"].get(name):
                raise CompilationError("scope edit lost its original source identity")
            component_edits.setdefault(name, []).append(edit)
            if edit not in edits.setdefault(name, []):
                edits[name].append(edit)
        for name, descriptor in report["sources"].items():
            original = Path(name).read_text()
            if _digest(original) != descriptor["sha256"] or descriptor["sha256"] != inputs[name]:
                raise CompilationError("original source changed before scope composition: " + name)
            replacement = descriptor["replacement"]
            if artifacts.get(replacement) != _apply(original, component_edits.get(name, [])):
                raise CompilationError("component source replacement disagrees with its original edit ledger")
            if name in sources and sources[name] != descriptor:
                raise CompilationError("scope roots disagree on their source replacement identity")
            originals[name], sources[name] = original, descriptor
        replacements = {item["replacement"] for item in report["sources"].values()}
        for name, content in artifacts.items():
            if name in replacements or name == "scope-manifest.json":
                continue
            _artifact_path(name)
            if name in outputs and outputs[name] != content:
                raise CompilationError("scope roots contain incompatible generated artifacts: " + name)
            outputs[name] = content
        for item in report["build_sources"]:
            name = item["path"]
            if name in build_sources and build_sources[name] != item:
                raise CompilationError("scope roots disagree on a build source contract: " + name)
            build_sources[name] = item
        manifests.append(report)
    for name, original in originals.items():
        outputs[sources[name]["replacement"]] = _apply(original, edits.get(name, []))
    for name, digest in inputs.items():
        if sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise CompilationError("original input changed before scope composition: " + name)
    provenance = {**manifests[0]["runtime_provenance"]}
    records = {}
    for item in manifests:
        for record in item["runtime_provenance"]["records"]:
            identity = record["id"]
            if identity in records and records[identity] != record:
                raise CompilationError("scope roots disagree on a runtime provenance identity")
            records[identity] = record
    provenance.update(records=list(records.values()), available=any(
        item["runtime_provenance"]["available"] for item in manifests))
    report = {
        "schema_version": 1,
        "entry": [item["entry"] for item in manifests],
        "scope_execution": "graph",
        "composition": {
            "schema_version": 1,
            "source": "same original bytes; checked disjoint edit union",
            "runtime": "one compiled memory authority; independent invocation contexts",
            "proof_scope": "component_manifests; original checked plans are not reissued under an aggregate entry",
            "components": [
                {"entry": item["entry"], "manifest_sha256": _digest(json.dumps(item, sort_keys=True))}
                for item in manifests
            ],
        },
        "component_manifests": manifests,
        "scope_count": sum(item["scope_count"] for item in manifests),
        "automatic_scope_available": any(item["automatic_scope_available"] for item in manifests),
        "automatic_estimate_available": any(item["automatic_estimate_available"] for item in manifests),
        "sources": sources,
        "source_inputs": inputs,
        "runtime": runtime,
        "source_edits": [edit for values in edits.values() for edit in values],
        "scopes": [scope for item in manifests for scope in item["scopes"]],
        "boundaries": [boundary for item in manifests for boundary in item["boundaries"]],
        "runtime_provenance": provenance,
        "build_sources": list(build_sources.values()),
        "artifacts_sha256": {name: _digest(content) for name, content in outputs.items()},
    }
    outputs["scope-manifest.json"] = json.dumps(report, indent=2) + "\n"
    return outputs, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    components = []
    for directory in args.components:
        report = json.loads((directory / "scope-manifest.json").read_text())
        artifacts = {name: (directory / str(_artifact_path(name))).read_text() for name in report["artifacts_sha256"]}
        components.append((artifacts, report))
    outputs, _report = compose_graph_scopes(components)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        destination = args.output_dir / str(_artifact_path(name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)


if __name__ == "__main__":
    main()
