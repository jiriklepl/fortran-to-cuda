"""Small compiler-owned identities for optional actual submission diagnostics."""

import json
from hashlib import sha256


class Provenance:
    def __init__(self):
        self.records = {}

    def add(self, kind, **facts):
        record = {"kind": kind, **facts}
        identity = sha256(json.dumps({"schema_version": 1, **record}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
        self.records[identity] = {"id": identity, **record}
        return identity

    def public(self, *, available=True, additional=()):
        records = {**self.records}
        for mapping in additional:
            for record in mapping.get("records", ()):
                records[record["id"]] = record
        return {"schema_version": 1, "available": available, "activation": "FORT_RUNTIME_TRACE=1",
                "runtime_api": ["fort_scope_trace_set_v1", "fort_scope_trace_restore_v1"],
                "records": [records[key] for key in sorted(records)], "unknown": "unknown",
                "semantics": "actual API submission and completed coherence commits; no physical execution timestamps",
                "batch_worker_attribution": "unavailable unless the individual implementation is explicitly tagged",
                "initial_context_creation": "unknown until the original source coordinator binds its owner",
                "execution_error_attribution": "unavailable; existing numerical error reporting is unchanged"}


def source_id(builder, kind, *, span=None, **facts):
    routine = builder.entry
    return builder.runtime_provenance.add(kind, procedure=routine.qualified,
        source=str(routine.scope.path), source_sha256=builder.analysis.sources[str(routine.scope.path)],
        **({"first_line": span[0], "last_line": span[1]} if span is not None else {}), **facts)


def position(builder, *, context="fort_context", owner=None, segment=None, operation="", implementation="",
             boundary="", previous="c_null_ptr", integer_kind="c_int32_t", nul="c_null_char",
             setter="fort_scope_trace_set_v1"):
    procedure = source_id(builder, "procedure")
    mask = 2 + 8 + 16 + 32 + (1 if owner is not None else 0) + (4 if segment is not None else 0)
    arguments = [context, f"{mask}_{integer_kind}", *[f"'{value or ''}'//{nul}" for value in
                           (owner, procedure, segment, operation, implementation, boundary)], previous]
    return [f"call {setter}( &",
            *["    " + argument + (", &" if index < len(arguments)-1 else ")")
              for index, argument in enumerate(arguments)]]


def call_frame(builder, context, operation, lines, *, implementation=""):
    """Save/restore a source call without shadowing its original visible names."""
    from compiler.ir import CompilationError

    names = {stem: f"fort_trace_{stem}_{operation[:12]}" for stem in
             ("saved", "state", "i32", "nul", "loc", "set", "restore")}
    scope = builder.entry.scope
    for name in names.values():
        if builder.analysis._binding(scope, name) or builder.analysis._candidates(scope, name):
            raise CompilationError("source call conflicts with diagnostic helper name: " + name)
    return ["block",
            f"use iso_c_binding, only: {names['i32']} => c_int32_t, {names['nul']} => c_null_char, &",
            f"    {names['loc']} => c_loc",
            f"use fort_scoped_memory, only: {names['state']} => fort_scope_trace_state_v1, &",
            f"    {names['set']} => fort_scope_trace_set_v1, {names['restore']} => fort_scope_trace_restore_v1",
            f"type({names['state']}), target :: {names['saved']}",
            *position(builder, context=context, operation=operation, implementation=implementation,
                      previous=f"{names['loc']}({names['saved']})", integer_kind=names['i32'],
                      nul=names['nul'], setter=names['set']), *lines,
            f"call {names['restore']}({context}, {names['saved']})", "end block"]


def borrowed(owner, nodes, lines):
    """Restore parent attribution even when a borrowed body closes its context.

    The wrapper neither reads entry-only dummies on the native ABI nor restores
    numerical state. A closed/stale handle is deliberately harmless to tracing.
    """
    from compiler.scopes.segments import statement_span

    first, last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
    operation = source_id(owner.builder, "borrowed_call", span=(first, last))
    result = call_frame(owner.builder, owner.context, operation, lines)
    return [f"if ({owner.control_guard}) then", *result, "else", *lines, "endif"] if owner.control_guard else result
