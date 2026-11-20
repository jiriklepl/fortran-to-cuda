"""Closed source/policy inputs separated from mutable generation work.

A context reads only the sealed session and immutable policy values. Replacing
any supported policy input discards the analysis's context; old handles then
fail rather than silently following a mutable analysis object. Parser/semantic
copies remain private projections and cannot acquire original membership.
"""

from compiler.frontend.proof_arena import DerivationArena


def fork_derivations(analysis):
    """A transaction/fork may read parent facts but cannot publish handles back."""
    context = getattr(analysis, "_source_authority_context", None)
    arena = getattr(analysis, "_source_derivation_arena", None)
    if context is not None and type(arena) is DerivationArena and arena._context is context:
        object.__setattr__(analysis, "_source_derivation_arena", arena.fork())
