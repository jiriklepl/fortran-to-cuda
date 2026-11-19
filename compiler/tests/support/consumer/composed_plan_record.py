"""Read compiler compositions without reissuing the constituent checked plans."""
import hashlib
import json

from .checked_execution_plan import _digest, _require, _text

_FIELDS = {"schema_version", "entry", "scope_execution", "composition", "component_manifests",
           "scope_count", "automatic_scope_available", "automatic_estimate_available", "sources",
           "source_inputs", "runtime", "source_edits", "scopes", "boundaries", "runtime_provenance",
           "build_sources", "artifacts_sha256"}
_POLICY = {
    "schema_version": 1,
    "source": "same original bytes; checked disjoint edits, TARGET-only union and named generated ENTRY exports",
    "runtime": "one compiled memory authority; independent invocation contexts",
    "proof_scope": "component_manifests; original checked plans are not reissued under an aggregate entry",
}


def validate_composed_plans(manifest, *, capture_facts):
    from .checked_execution_plan import validate_checked_execution_plans
    from .original_worksharing_record import (
        validate_original_worksharing_package_constraints,
    )

    _require(set(manifest) == _FIELDS and type(manifest["schema_version"]) is int
             and manifest["schema_version"] == 1 and manifest["scope_execution"] == "graph",
             "composition envelope differs")
    _require(capture_facts is None, "composition requires each original root's capture identity")
    components, record = manifest["component_manifests"], manifest["composition"]
    _require(type(components) is list and bool(components) and type(record) is dict
             and set(record) == {*_POLICY, "components"}
             and type(record["schema_version"]) is int
             and all(record[key] == value for key, value in _POLICY.items()),
             "composition policy differs")
    _require(type(record["components"]) is list and len(record["components"]) == len(components),
             "composition component inventory differs")
    _require(type(manifest["scope_count"]) is int and all(type(manifest[key]) is bool
             for key in ("automatic_scope_available", "automatic_estimate_available")),
             "composition aggregate count or availability differs")
    inputs, sources, edits, builds, artifacts, provenance = {}, {}, {}, {}, {}, {}
    entries, scopes, boundaries = [], [], []
    for component, identity in zip(components, record["components"], strict=True):
        _require(type(component) is dict and "composition" not in component
                 and "component_manifests" not in component and _text(component.get("entry"))
                 and component.get("scope_execution") == "graph",
                 "composition requires non-nested original roots")
        _digest(component)  # Reject nonfinite/non-JSON content before the public hash recipe.
        expected = {"entry": component["entry"], "manifest_sha256": hashlib.sha256(
            json.dumps(component, sort_keys=True).encode()).hexdigest()}
        _require(identity == expected and component["entry"] not in entries,
                 "composition component identity differs")
        _require(component.get("runtime") == manifest["runtime"] and bool(manifest["runtime"]),
                 "composition runtime differs")
        validate_checked_execution_plans(component)
        entries.append(component["entry"])
        for key in ("source_inputs", "sources", "artifacts_sha256", "runtime_provenance"):
            _require(type(component.get(key)) is dict, "composition requires original " + key)
        for key in ("scopes", "source_edits", "boundaries", "build_sources"):
            _require(type(component.get(key)) is list, "composition requires original " + key)
        _require(type(component.get("scope_count")) is int and component["scope_count"] >= 0,
                 "composition scope count differs")
        for key in ("automatic_scope_available", "automatic_estimate_available"):
            _require(type(component.get(key)) is bool, "composition availability differs")
        for key, target in (("source_inputs", inputs), ("sources", sources)):
            for name, value in component[key].items():
                _require(name not in target or target[name] == value,
                         "composition conflicting original " + key)
                target[name] = value
        for edit in component["source_edits"]:
            _require(type(edit) is dict and _text(edit.get("source")), "composition source edit differs")
            group = edits.setdefault(edit["source"], [])
            if edit not in group:
                group.append(edit)
        for unit in component["build_sources"]:
            _require(type(unit) is dict and _text(unit.get("path")), "composition build source differs")
            name = unit["path"]
            _require(name not in builds or builds[name] == unit, "composition build contract differs")
            builds[name] = unit
        replacement_paths = {value["replacement"] for value in component["sources"].values()}
        for name, value in component["artifacts_sha256"].items():
            if name in replacement_paths:
                continue
            _require(name not in artifacts or artifacts[name] == value, "composition helper artifact differs")
            artifacts[name] = value
        records = component["runtime_provenance"].get("records")
        _require(type(records) is list, "composition runtime provenance differs")
        for item in records:
            _require(type(item) is dict and _text(item.get("id")), "composition provenance identity differs")
            name = item["id"]
            _require(name not in provenance or provenance[name] == item,
                     "composition conflicting runtime provenance")
            provenance[name] = item
        scopes.extend(component["scopes"])
        boundaries.extend(component["boundaries"])
    expected_provenance = {**components[0]["runtime_provenance"], "records": list(provenance.values()),
                           "available": any(item["runtime_provenance"]["available"] for item in components)}
    expected = {"entry": entries, "source_inputs": inputs, "sources": sources,
                "source_edits": [edit for group in edits.values() for edit in group],
                "build_sources": list(builds.values()), "scopes": scopes, "boundaries": boundaries,
                "runtime_provenance": expected_provenance,
                "scope_count": sum(item["scope_count"] for item in components),
                **{key: any(item[key] for item in components)
                   for key in ("automatic_scope_available", "automatic_estimate_available")}}
    _require(all(manifest[key] == value for key, value in expected.items()),
             "composition aggregate differs from its original components")
    replacements = {value["replacement"] for value in sources.values()}
    _require(type(manifest["artifacts_sha256"]) is dict
             and set(manifest["artifacts_sha256"]) == set(artifacts) | replacements
             and all(manifest["artifacts_sha256"][name] == value for name, value in artifacts.items()),
             "composition artifacts differ from their original components")
    validate_original_worksharing_package_constraints(components)
