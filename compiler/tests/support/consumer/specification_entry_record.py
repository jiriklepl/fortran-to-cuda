"""Validate the pre-entry observation phase of an original checked CALL."""
from hashlib import sha256

from .checked_execution_plan import _require, _sha256, _text
from .conditional_return_record import _context_member, _false, _keys, _sealed, _slot
from .native_opaque_access_record import _storage
from .source_task_record import _binding

_UNSAFE = {"pointer", "target", "optional", "volatile", "asynchronous", "value", "codimension"}


def validate_runtime(runtime, returned):
    # Retain historical initialized-only artifacts for their original controls.
    # New runtimes publish physical versions independently of definition proof.
    capability = runtime.get("specification_entry", {}) if isinstance(runtime, dict) else {}
    version = capability.get("abi_version") if isinstance(capability, dict) else None
    _require(type(version) is int and version in {2, 3}, "specification entry runtime ABI differs")
    _require(isinstance(runtime, dict) and runtime.get("specification_entry") == {
        "abi_version": version, "observe": "fort_scope_observe_v1", "fallback": "fort_scope_observe_all_v1",
        "handle_check": "fort_scope_buffer_valid_v1",
        "regions": ("initialized mapped metadata regions; whole domain if mappings or live handles are unavailable"
                    if version == 2 else
                    "tracked physical metadata regions without definition authority; whole domain if mappings or live handles are unavailable"),
        "replicas": "read-only publication preserves current device replicas and generations",
        "failure": "stop before original entry; never replay"}, "specification entry runtime ABI differs")
    _require(any("specification_reads" in member.get("entry_exit", {}) for member in returned["members"]),
             "specification entry policy lacks its original child projection")


def validate_projection(proof, entry, returned, scope):
    _require(returned is not None and returned["version"] == 5
             and returned.get("specification_entry_policy_version") == 2
             and entry["procedure"] != scope["checked_execution_plan"]["entry"],
             "specification entry requires its distinct checked child policy")
    _sealed(proof, {"identity", "version", "kind", "procedure", "source_inputs", "original_procedure_sha256",
                    "declarations", "array_resources", "phase", "bounds_evaluation", "region_authority",
                    "requires_caller_association_and_coherence", "definition_authority", "alias_authority",
                    "lifetime_authority", "body_completion_authority", "payload_access_authorized", "execution_authorized"})
    _require(type(proof["version"]) is int and proof["version"] == 1
             and proof["kind"] == "original_call_free_specification_reads"
             and proof["procedure"] == entry["procedure"] and proof["source_inputs"] == returned["source_inputs"]
             and proof["original_procedure_sha256"] == entry["original_body_sha256"]
             and proof["phase"] == "before original procedure entry and specification evaluation"
             and proof["bounds_evaluation"] == "original Fortran only; this projection does not evaluate expressions"
             and proof["region_authority"] == "storage may-read projection; no subscript bounds or exact region proof"
             and proof["requires_caller_association_and_coherence"] is True,
             "specification entry source projection differs")
    _false(proof, ("definition_authority", "alias_authority", "lifetime_authority", "body_completion_authority",
                   "payload_access_authorized", "execution_authorized"))
    roots = proof["array_resources"]
    _require(isinstance(roots, list) and bool(roots) and all(_text(root) for root in roots)
             and roots == sorted(set(roots)) and isinstance(proof["declarations"], list)
             and bool(proof["declarations"]), "specification entry has no original array-read inventory")
    resources, declarations = set(), set()
    for node in proof["declarations"]:
        _keys(node, {"kind", "source", "items", "declaration"})
        _require(node["kind"] == "declaration" and _text(node["source"]) and isinstance(node["items"], list),
                 "specification entry requires original declaration reads only")
        binding = node["declaration"]
        _require(isinstance(binding, dict) and binding.get("type") in {"integer", "real", "logical"}
                 and type(binding.get("kind")) is int and binding["kind"] > 0
                 and type(binding.get("rank")) is int and binding["rank"] >= 0,
                 "specification entry original numeric declaration differs")
        # Reuse descriptor structure without imposing the numerical worker's
        # type/rank restrictions on untouched original Fortran declarations.
        _binding({**binding, "type": "integer", "kind": 4}, max_rank=binding["rank"])
        _require(not set(binding["attributes"]) & _UNSAFE and binding["resource"] not in declarations
                 and len(binding["shape"]) == len(binding["lower_bounds"]) == binding["rank"],
                 "specification entry original declaration aliases or dimensions differ")
        declarations.add(binding["resource"])
        for access in node["items"]:
            _keys(access, {"storage", "kind", "procedure", "source", "external"})
            _require(access["kind"] == "read" and access["procedure"] == proof["procedure"]
                     and _text(access["source"]) and type(access["external"]) is bool,
                     "specification entry cannot acquire body or write authority")
            storage = access["storage"]
            _storage(storage)
            _require(storage["type"] == "integer" and not storage["component_path"]
                     and storage["base_resource"] == storage["resource"] and not set(storage["attributes"]) & _UNSAFE,
                     "specification entry requires independent original integer storage")
            if storage["rank"]:
                resources.add(storage["resource"])
    _require(roots == sorted(resources), "specification entry array inventory differs from original reads")


def validate_call(route, procedure, procedures, coordinators, scope, source_edits, returned):
    entries = {member["procedure"]: member["entry_exit"] for member in returned["members"]}
    proof = entries[route["procedure"]].get("specification_reads")
    value = route.get("specification_entry")
    if proof is None:
        _require(value is None, "specification entry publication has no original read projection")
        return
    validate_projection(proof, entries[route["procedure"]], returned, scope)
    _keys(value, {"version", "proof_identity", "context", "source_span", "publication_sha256", "runtime", "policy", "bindings",
                  "replicas", "failure"})
    caller = procedure["procedure"]
    namespace = _context_member(caller, scope, coordinators)
    context = (scope["owner"] if namespace == scope["checked_execution_plan"]["entry"]
               else "fort_child_context_" + sha256(namespace.encode()).hexdigest()[:12])
    expected = []
    if all(root in route["canonical_resources"] for root in proof["array_resources"]):
        expected = [{"resource": root, "callee_slot": _slot(route["procedure"], root),
                     "caller_slot": route["handle_forwarding"][str(_slot(route["procedure"], root))],
                     "canonical_resource": route["canonical_resources"][root]}
                    for root in proof["array_resources"]]
    _require(isinstance(value["bindings"], list) and value["bindings"] == expected
             and all(type(row.get(key)) is int for row in value["bindings"] for key in ("callee_slot", "caller_slot")),
             "specification entry metadata differs from checked original CALL forwarding")
    runtime = "fort_scope_observe_v1" if expected else "fort_scope_observe_all_v1"
    _require(type(value["version"]) is int and value["version"] == 2 and value["proof_identity"] == proof["identity"]
             and value["context"] == context and value["runtime"] == runtime
             and value["policy"] == "publish mapped metadata; observe the domain when mappings or live handles are unavailable"
             and value["replicas"] == "preserve valid device replicas and allocation generations"
             and value["failure"] == "stop before child entry; never replay" and _sha256(value["publication_sha256"]),
             "specification entry coherence contract differs")
    span = value["source_span"]
    _require(isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span) and 1 <= span[0] <= span[1],
             "specification entry original CALL span differs")
    calls = [call for call in procedures[route["procedure"]]["calls"] if call.get("caller") == caller
             and [call.get("first_line"), call.get("last_line")] == span
             and call.get("handle_bindings") == route["handle_forwarding"]]
    _require(len(calls) == 1 and len(route["selected_node_ids"]) == 1,
             "specification entry is detached from the original checked CALL")
    _require(isinstance(source_edits, list), "specification entry requires its emitted source edit")
    edits = [edit for edit in source_edits if edit.get("source") == procedure["source"]
             and [edit.get("first_line"), edit.get("last_line")] == span]
    _require(len(edits) == 1 and edits[0].get("source_sha256") == procedure["source_sha256"]
             and isinstance(edits[0].get("replacement"), str), "specification entry source edit differs")
    publication = "\n".join(["block", "use iso_c_binding, only: c_int",
        "use fort_scoped_memory, only: fort_scope_observe_all_v1, FORT_SCOPE_OK",
        "integer(c_int) :: fort_specification_status", f"if ({context} /= 0) then",
        f"fort_specification_status = fort_scope_observe_all_v1({context})",
        "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification publication failed'",
        "endif", "end block", ""])
    if expected:
        slots = sorted({row["caller_slot"] for row in expected})
        publication = "\n".join(["block", "use iso_c_binding, only: c_int, c_int64_t, c_size_t, c_loc",
            "use fort_scoped_memory, only: fort_scope_observe_v1, fort_scope_observe_all_v1, &",
            "fort_scope_buffer_valid_v1, FORT_SCOPE_OK",
            "integer(c_int) :: fort_specification_status, fort_specification_valid",
            "integer :: fort_specification_index", "logical :: fort_specification_precise",
            f"integer(c_int64_t), target :: fort_specification_buffers({len(slots)})",
            f"if ({context} /= 0) then",
            *[f"fort_specification_buffers({index}) = fort_buffer_{slot}" for index, slot in enumerate(slots, 1)],
            "fort_specification_precise = .true.", f"do fort_specification_index = 1, {len(slots)}",
            f"fort_specification_status = fort_scope_buffer_valid_v1({context}, &",
            "fort_specification_buffers(fort_specification_index), fort_specification_valid)",
            "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification handle check failed'",
            "if (fort_specification_valid == 0) fort_specification_precise = .false.", "enddo",
            "if (fort_specification_precise) then",
            f"fort_specification_status = fort_scope_observe_v1({context}, &",
            f"c_loc(fort_specification_buffers), {len(slots)}_c_size_t)", "else",
            f"fort_specification_status = fort_scope_observe_all_v1({context})", "endif",
            "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification publication failed'",
            "endif", "end block", ""])
    text = edits[0]["replacement"]
    marker = "call " + coordinators[route["procedure"]]["entry"] + "("
    _require(value["publication_sha256"] == sha256(publication.encode()).hexdigest()
             and text.count(publication) == 1 and text.find(marker) > text.find(publication),
             "specification entry must publish before the original checked CALL")
