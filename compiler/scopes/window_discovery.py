"""Bounded discovery of one-shot owners from exact original source windows.

Discovery supplies neither storage facts nor numerical authority. Each attempt
uses the registered source-position proof and ordinary reached emission in an
isolated generation transaction. Failed attempts leave ordinary generation
unchanged. Original module allocations require explicit selected-window facts
and a separate registered descriptor proof; local allocations remain native.
"""

from __future__ import annotations

from compiler.frontend.reached_windows import prove_reached_window
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner

DISCOVERY_VERSION = 2
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
    return sum(bool(member.builder.closure(call.procedure)[0])
               for member in owner.members.values() for unit in member.units for call in unit.scope.calls)


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
                proof = prove_reached_window(candidate.analysis, candidate.entry.qualified,
                                             originals, tuple(roots), candidate.facts)
                owner = ReachedWindowOwner(candidate, proof, originals)
                if not owner.run():
                    raise CompilationError("selected source window has no admitted numerical implementation")
                calls = _numerical_calls(owner)
                if calls < 2:
                    raise CompilationError("selected source window needs two admitted numerical invocations")
                if any(member.boundaries for member in owner.members.values()):
                    raise CompilationError("selected source window contains an ownership-ending reached operation")
                scope = owner.builder.scopes[-1]
                if scope.get("automatic_preflight", {}).get("selection") == "native":
                    raise CompilationError("selected window has no available automatic GPU alternative: "
                                           + scope["automatic_preflight"]["reason"])
                attempt.update({"accepted": True, "proof_identity": proof.identity,
                                "numerical_invocations": calls})
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
