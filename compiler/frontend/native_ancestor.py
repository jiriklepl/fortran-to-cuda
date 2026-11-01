"""Source-bound selection of device subtrees under native graph ancestors.

The request restricts discovery. It never supplies a worker, call association,
payload effect, lifetime, definition, or permission to keep a context live.
Every selected descendant still passes ordinary checked task admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from compiler.frontend.source_functions import _hash, _original
from compiler.frontend.structured_effects import _freeze_source_record, _thaw
from compiler.ir import CompilationError

_ISSUER = object()


def _derive(analysis, entry, request):
    analysis.verify_source()
    local = (isinstance(request, dict) and type(request.get("version")) is int and request["version"] == 2
             and request.get("lifetime") == "owner_local_call_groups")
    supported = (isinstance(request, dict) and type(request.get("version")) is int and request["version"] == 4
                 and request.get("lifetime") == "supported_original_path")
    keys = ({"version", "device_roots", "lifetime", "path"} if supported else
            {"version", "device_roots", "lifetime"} if local else {"version", "device_roots"})
    if (not isinstance(request, dict) or set(request) != keys
            or type(request["version"]) is not int or request["version"] != (4 if supported else 2 if local else 1)
            or not isinstance(request["device_roots"], list) or not request["device_roots"]
            or len(request["device_roots"]) > analysis.procedure_limit
            or any(not isinstance(name, str) or name != name.lower() for name in request["device_roots"])
            or len(set(request["device_roots"])) != len(request["device_roots"])):
        raise CompilationError("native ancestor requires a versioned exact original device-root selection")
    root = _original(analysis, entry)
    if root.source_kind != "module" or entry in request["device_roots"]:
        raise CompilationError("native ancestor requires a module entry separate from selected descendants")
    members = []
    for name in sorted(request["device_roots"]):
        routine = _original(analysis, name)
        if routine.source_kind != "module":
            raise CompilationError("native ancestor device root requires original module procedure authority")
        members.append({"procedure": name, "source": str(routine.scope.path),
                        "source_sha256": analysis.sources[str(routine.scope.path)],
                        "original_body_sha256": _hash(str(routine.scope.node))})
    if supported:
        from compiler.frontend.supported_paths import requested_supported_path
        requested_supported_path(analysis, root, request["path"], tuple(request["device_roots"]))
    analysis.verify_publication()
    return {"version": 4 if supported else 2 if local else 1, "kind": "original_native_ancestor_selection", "entry": entry,
            **({"lifetime": "supported_original_path", "path_request": request["path"]} if supported else
               {"lifetime": "owner_local_call_groups"} if local else {}),
            "source_inputs": analysis.inputs.identity(), "original_entry_body_sha256": _hash(str(root.scope.node)),
            "device_roots": members,
            "selection_semantics": "only exact selected original module subtrees may undergo device discovery; all ancestors remain original native source",
            "call_authority": "independent exact original call association and checked companion admission",
            "native_execution": "existing precise native effects and coherence hooks or reached publication and full domain retirement",
            "unselected_native_once": True, "payload_access_authorized": False,
            "descriptor_stability_authorized": False, "entry_definition_authority": False,
            "execution_authorized": False, "placement_authorized": False, "alias_authority": False}


@dataclass(frozen=True, slots=True)
class NativeAncestorSelection:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _entry: str
    _record: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis=None):
        analysis = self._analysis if analysis is None else analysis
        if (analysis is not self._analysis or self._issuer is not _ISSUER
                or getattr(analysis, "_native_ancestor_selections", {}).get(self.identity) is not self):
            raise CompilationError("native ancestor requires its exact registered original selection")
        document = analysis.contracts.get(self._entry, {})
        request = document.get("native_ancestor") if isinstance(document, dict) else None
        record = _derive(analysis, self._entry, request)
        if _hash(record) != self.identity or _freeze_source_record(analysis, record) != self._record:
            raise CompilationError("native ancestor source selection changed")
        return self

    def selects(self, routine):
        self.validate()
        if _original(self._analysis, routine.qualified) is not routine:
            raise CompilationError("native ancestor requires exact original procedure identity")
        return any(item["procedure"] == routine.qualified for item in self._record["device_roots"])


def native_ancestor_selection(analysis, entry):
    document = analysis.contracts.get(entry, {})
    if not isinstance(document, dict) or "native_ancestor" not in document:
        return None
    record = _derive(analysis, entry, document["native_ancestor"])
    identity = _hash(record)
    registry = getattr(analysis, "_native_ancestor_selections", None)
    if registry is None:
        registry = analysis._native_ancestor_selections = {}
    if identity not in registry:
        registry[identity] = NativeAncestorSelection(identity, analysis, entry, _freeze_source_record(analysis, record), _ISSUER)
    return registry[identity].validate(analysis)
