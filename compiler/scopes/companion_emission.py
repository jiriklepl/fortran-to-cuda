"""One exact source definition for a checked original module companion.

Calls and their storage mappings remain independently checked. This ledger only
owns the helper's source edits; different definitions never merge by text.
"""
import json
from collections import Counter

from compiler.ir import CompilationError


def publish_original_companion(member, edits):
    from compiler.scopes.execution_plan import _configuration
    from compiler.scopes.module_owner import ModuleOwner

    if not isinstance(member, ModuleOwner):
        raise CompilationError("original companion publication requires its checked module owner")
    owner = member.owner
    plan = owner.execution_plan.validate()
    if not any(item is member for item in plan.members):
        raise CompilationError("original companion publication lost its checked plan member")
    builder, routine = owner.builder, member.routine
    path = routine.scope.path
    edits = tuple(edits)
    if any(source != path or type(text) is not str or type(prepend) is not bool
           for source, _first, _last, text, prepend in edits):
        raise CompilationError("original companion publication changed its source edit ownership")
    signature = json.dumps({
        "procedure": routine.qualified, "source": str(path),
        "source_sha256": builder.analysis.sources[str(path)],
        "graph_identity": member.graph.identity,
        "configuration": _configuration(member.builder),
        "arguments": [routine.scope.bindings[name].public() for name in routine.arguments],
        "controls": {name: getattr(member, name, None) for name in (
            "entry", "kind", "context", "enabled", "epoch_pristine", "query_plan",
            "control_guard", "retention_allowed", "eligibility_allowed")},
        "parameterized_storage_identity": member.resource_slot_namespace is member,
        "handle_numbers": list(member.handle_numbers()),
        "resource_slot_roots": dict(member.resource_slot_roots),
        "body_input_intents": sorted(member.body_input_intents),
    }, sort_keys=True, separators=(",", ":"))
    # The emitted symbol has one original owner, even if a name collision is
    # introduced between otherwise different original procedures.
    key = str(path), member.entry
    record = signature, edits
    previous = builder._module_companion_publications.get(key)
    if previous is not None:
        if previous != record:
            raise CompilationError("shared original module companion has conflicting source, signature or controls: "
                                   + routine.qualified)
        available = Counter(builder.edits.get(path, ()))
        required = Counter((first, last, text) for _path, first, last, text, _prepend in edits)
        if any(available[edit] < count for edit, count in required.items()):
            raise CompilationError("shared original module companion lost its published source edits")
        return False
    for source, first, last, text, prepend in edits:
        builder.add_edit(source, first, last, text, prepend=prepend)
    builder._module_companion_publications[key] = record
    return True
