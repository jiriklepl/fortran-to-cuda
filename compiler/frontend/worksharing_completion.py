"""Original-team participation authority for one synchronized worksharing DO.

This proof permits numerical analysis of a subregion. It cannot authorize
standalone execution, native coherence hooks, or GPU legality by itself.
"""

from dataclasses import dataclass, field
from hashlib import sha256
import json

from compiler.frontend.native_completion import _directive, _joined_completion_facts
from compiler.ir import CompilationError

WORKSHARING_COMPLETION_VERSION = 1


@dataclass(frozen=True)
class WorksharingCompletionProof:
    procedure: str
    structured_identity: str
    identity: str
    selected_node_ids: tuple[str, ...]
    private_roots: tuple[str, ...]
    parent: object = field(repr=False, compare=False)
    opening: object = field(repr=False, compare=False)
    loop: object = field(repr=False, compare=False)
    ending: object = field(repr=False, compare=False)

    def validate(self, analysis, procedure, selected):
        graph, identities = analysis._selected_source(procedure, selected)
        if (procedure != self.procedure or graph.identity != self.structured_identity
                or identities != self.selected_node_ids
                or analysis._worksharing_completions.get(self.identity) is not self):
            raise CompilationError('worksharing subregion lacks registered original-team authority')
        self.parent.validate(analysis, procedure, self.parent.selected_node_ids)
        return self

    def public(self):
        return {'schema_version': WORKSHARING_COMPLETION_VERSION,
                'proof_role': 'original_team_worksharing_completion', 'proof_identity': self.identity,
                'available': True,
                'procedure': self.procedure, 'structured_identity': self.structured_identity,
                'parent_completion_identity': self.parent.identity,
                'selected_original_nodes': list(self.selected_node_ids),
                'private_resources': list(self.private_roots),
                'caller_contract': 'all members of the original team at the original reached DO',
                'join': 'explicit original END DO' if self.ending is not None else 'implicit original worksharing completion',
                'standalone_execution_authority': False, 'native_effects_authority': False,
                'gpu_legality_established': False}


def prove_worksharing_completion(analysis, procedure, joined, selected):
    """Resolve an original DO within a complete, uniformly reached team.

    NOWAIT anywhere in the selected team prevents splitting: an earlier native
    unit may still be running when a later unit reaches its coordinator.
    Combined PARALLEL DO continues to use the whole-group analysis path.
    """
    parent = analysis.joined_completion(procedure, joined)
    units = []
    _joined_completion_facts(analysis, procedure, joined, worksharing=units)
    if not units:
        raise CompilationError('worksharing subregion requires a separate original PARALLEL and DO')
    if any(_directive(ending) == 'end do nowait' for _, _, ending in units if ending is not None):
        raise CompilationError('worksharing subregions require completion at every original DO; NOWAIT remains native')
    graph, identities = analysis._selected_source(procedure, selected)
    matches = [(opening, loop, ending) for opening, loop, ending in units
               if analysis._selected_source(procedure, (loop,))[1] == identities]
    if len(matches) != 1:
        raise CompilationError('worksharing selection must be exactly one associated original DO')
    opening, loop, ending = matches[0]
    record = {'version': WORKSHARING_COMPLETION_VERSION, 'parent': parent.identity,
              'selected': identities, 'barrier': 'explicit' if ending is not None else 'implicit'}
    identity = sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return WorksharingCompletionProof(procedure, graph.identity, identity, identities,
                                      parent.private_roots, parent, opening, loop, ending)
