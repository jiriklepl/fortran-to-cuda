"""Bounded discovery of one-shot owners from exact original source windows.

Discovery supplies neither storage facts nor numerical authority. Each attempt
uses the registered source-position proof and ordinary reached emission in an
isolated generation transaction. Failed attempts leave ordinary generation
unchanged. Original module allocations require explicit selected-window facts
and a separate registered descriptor proof; local allocations remain native.
"""

from __future__ import annotations

from compiler.frontend.reached_windows import plan_reached_window
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner

DISCOVERY_VERSION = 5
MAX_ADJACENT_UNITS = 4
MAX_ATTEMPTS = 16


def _candidates(graph):
    """Yield immediate original sequences, never a cut through a construct."""
    root = graph.nodes[graph.root_id]
    sequences = []
    # First-traversal windows let excluded allocation-changing loop tails stay
    # native. Only the proof provider can admit the original counted header.
    for identity in root.children:
        node = graph.nodes[identity]
        if node.kind == "loop" and node.children:
            body = graph.nodes[node.children[-1]]
            if body.kind == "sequence":
                sequences.append(body)
    sequences.append(root)
    for sequence in sequences:
        for first in range(len(sequence.children) - 1):
            maximum = min(MAX_ADJACENT_UNITS, len(sequence.children) - first)
            for width in range(maximum, 1, -1):
                identities = sequence.children[first:first + width]
                originals = tuple(original for identity in identities for original in graph.source_nodes(identity))
                if originals:
                    yield identities, originals


def _numerical_calls(owner):
    """Count admitted numerical invocations, including repeated shared leaves."""
    direct = {name: sum(bool(member.builder.closure(call.procedure)[0])
                       for unit in member.units for call in unit.scope.calls)
              for name, member in owner.members.items()}
    if not any(getattr(member.builder, "_mapped_reached_child", None) is not None
               for member in owner.members.values()):
        return sum(direct.values())
    edges, seen, visits = {}, set(), 0
    for name, member in owner.members.items():
        for site in getattr(member, "call_sites", ()):
            identity = site.get("mapped_call_node_id")
            if identity is None:
                continue
            caller = owner.members.get(site["caller"])
            span = site["first_line"], site["last_line"]
            if caller is None or getattr(caller, "mapped_call_patches", {}).get(span) != identity:
                raise CompilationError("numerical invocation count lacks its retained original call site")
            key = site["caller"], identity
            if key in seen:
                raise CompilationError("numerical invocation count repeats one original call site")
            seen.add(key)
            visits += 1
            if visits > owner.builder.analysis.procedure_limit:
                raise CompilationError("numerical invocation call traversal budget exhausted")
            edges.setdefault(site["caller"], []).append(name)
    counts = {}

    def count(name, active=()):
        if name in active or len(active) >= owner.builder.analysis.depth_limit:
            raise CompilationError("numerical invocation call depth or recursion budget exhausted")
        if name not in counts:
            counts[name] = direct[name] + sum(count(child, (*active, name)) for child in edges.get(name, ()))
        return counts[name]

    return count(owner.routine.qualified)


def _source_declines(owner):
    """Retain exact proof reasons when an otherwise valid window is unused."""
    proof = owner.window_proof.validate(owner.builder.analysis, owner.routine.qualified,
                                        owner.window_originals)
    public = proof.public()
    points = [("prefix_disable", point) for point in public["prefix_eligibility"]["disable_points"]]
    points.extend(("active_close", point) for point in public["planned_closures"]["close_points"])
    if len(points) > owner.builder.analysis.operation_limit:
        raise CompilationError("original-window decline diagnostic budget exhausted")
    return [{"phase": phase, **{field: point[field] for field in
                              ("node_id", "selected_node_ids", "span", "original_source_kind", "reason")},
             "completion_identity": point.get("completion_identity")}
            for phase, point in points]


def _no_implementation(owner, attempt, reason):
    declines = _source_declines(owner)
    attempt["source_declines"] = declines
    details = "; ".join(f"{point['phase']} {point['original_source_kind']} at {point['span']}: {point['reason']}"
                        for point in declines)
    return CompilationError(reason + (": " + details if details else ""))


def discover_reached_window(builder):
    """Adopt a useful bounded window, or leave ordinary generation intact.

    This runs before inline preparation. An incomplete whole-entry descriptor
    proof is a trigger, never admission of a shorter descriptor lifetime.
    """
    formals = {binding.root for binding in builder.entry.scope.bindings.values()
               if binding.name in builder.entry.arguments and binding.rank and "allocatable" in binding.attributes}
    modules = sorted(binding.root for module in builder.analysis.modules.values()
                     for binding in module.bindings.values() if binding.rank and "allocatable" in binding.attributes
                     and isinstance(fact := builder.facts["captures"].get(binding.root), dict)
                     and fact.get("lifetime") == "selected_window")
    if not formals and not modules:
        return False
    descriptor = builder.analysis.descriptor_stability(builder.entry.qualified)
    stable = {item["resource"] for item in descriptor["resources"] if item["stable"]}
    unstable = sorted(formals - stable)
    if not unstable and not modules:
        return False
    record = {
        "schema_version": DISCOVERY_VERSION,
        "policy": "bounded original immediate windows; one first traversal; no reopening",
        "trigger": {"whole_entry_unstable_allocatable_formals": unstable,
                    "selected_window_module_resources": modules},
        "limits": {"adjacent_source_units": MAX_ADJACENT_UNITS, "attempts": MAX_ATTEMPTS,
                   "minimum_numerical_invocations": 2},
        "selected": False, "attempts": [],
        "storage_authority": "independently admitted invocation facts plus registered selected-source proofs",
        "placement_authority": "existing compiler-owned offline cost decisions",
    }
    builder.reached_window_discovery = record
    graph = builder.analysis.structure(builder.entry.qualified)
    if not graph.available:
        record["reason"] = "bounded original source structure unavailable: " + "; ".join(graph.reasons)
        return False
    roots = []
    for root in sorted(builder.facts["captures"]):
        try:
            binding = builder.visible_binding(builder.entry, root)
            if binding.rank:
                builder.capture_fact(binding)
                roots.append(root)
        except CompilationError:
            # This cannot hide a dependency: the source-position provider
            # rejects every untracked payload array in the prefix/window.
            continue
    if not roots:
        record["reason"] = "no independently admitted original array captures"
        return False
    for identities, originals in _candidates(graph):
        if len(record["attempts"]) >= MAX_ATTEMPTS:
            record["reason"] = "bounded original-window search budget exhausted"
            break
        attempt = {"selected_node_ids": list(identities),
                   "first_line": graph.nodes[identities[0]].span[0],
                   "last_line": graph.nodes[identities[-1]].span[-1]}
        record["attempts"].append(attempt)
        try:
            with GenerationTransaction(builder) as transaction:
                candidate = transaction.builder
                proof = plan_reached_window(candidate.analysis, candidate.entry.qualified,
                                            originals, tuple(roots), candidate.facts)
                owner = ReachedWindowOwner(candidate, proof, originals)
                if not owner.run():
                    raise _no_implementation(owner, attempt,
                                             "selected source window has no admitted numerical implementation")
                calls = _numerical_calls(owner)
                if calls < 2:
                    raise _no_implementation(owner, attempt,
                                             "selected source window needs two admitted numerical invocations")
                proof = owner.window_proof
                scope = owner.builder.scopes[-1]
                if scope.get("automatic_preflight", {}).get("selection") == "native":
                    raise CompilationError("selected window has no available automatic GPU alternative: "
                                           + scope["automatic_preflight"]["reason"])
                attempt.update({"accepted": True, "proof_identity": proof.identity,
                                "numerical_invocations": calls,
                                "planned_closures": proof.public()["planned_closures"]})
                scope["reached_window"]["automatic_discovery"] = True
                transaction.adopt(owner.builder)
        except CompilationError as error:
            attempt.update({"accepted": False, "reason": str(error)})
            continue
        record.update({"selected": True, "proof_identity": attempt["proof_identity"],
                       "selected_node_ids": list(identities)})
        # The adopted builder owns a detached diagnostic record. Publish the
        # final record only after successful transaction exit.
        builder.reached_window_discovery = record
        return True
    record.setdefault("reason", "no useful proved original window within the bounded search")
    return False
