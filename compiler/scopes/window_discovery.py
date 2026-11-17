"""Bounded discovery of one-shot owners from exact original source windows.

Discovery supplies neither storage facts nor numerical authority. Each attempt
uses the registered source-position proof and ordinary reached emission in an
isolated generation transaction. Failed attempts leave ordinary generation
unchanged. Original module allocations require explicit selected-window facts
and a separate registered descriptor proof; local allocations remain native.
"""

from __future__ import annotations


DISCOVERY_VERSION = 7
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
