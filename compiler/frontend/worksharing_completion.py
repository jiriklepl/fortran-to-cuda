"""Distinct numerical participation and native-coherence worksharing proofs.

Both require original synchronized work inside a complete joined team. Neither
proof authorizes standalone execution or establishes GPU legality by itself.
"""

from dataclasses import dataclass, field
from hashlib import sha256
import json

from fparser.two.utils import walk

from compiler.frontend.native_completion import _directive, _joined_completion_facts
from compiler.ir import CompilationError

WORKSHARING_COMPLETION_VERSION = 1
WORKSHARING_NATIVE_COMPLETION_VERSION = 1


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

    def following(self, analysis):
        """Retain original suffix liveness even if an outlining caller omits it.

        Use prepared source order here: included native statements may lack
        editable original spans, but their reads cannot disappear from proof.
        Taking later alternative branches too is deliberately conservative.
        """
        self.validate(analysis, self.procedure, (self.loop,))

        def span(node):
            positions = [item.item.span for item in walk(node) if getattr(item, 'item', None) is not None]
            if not positions or any(position is None for position in positions):
                raise CompilationError('worksharing liveness requires original prepared source ordering')
            return min(position[0] for position in positions), max(position[1] for position in positions)

        last = span(self.loop)[1]
        selected = {id(node) for node in walk(self.loop)}
        result = []

        def visit(node):
            if id(node) in selected:
                return
            low, high = span(node)
            if low > last:
                result.append(node)
            elif high > last:
                for child in getattr(node, 'content', ()):
                    visit(child)

        for node in analysis.routines[self.procedure].execution.content:
            visit(node)
        return tuple(result)


@dataclass(frozen=True)
class WorksharingNativeCompletionProof:
    """Native coherence authority at synchronized points in an original team.

    The selected loops and any selected uniform branch remain in their original
    lexical team. A coordinator must complete a host-begin hook before releasing
    that team into the selected work, and wait for its original completion before
    committing host writes. This token does not authorize numerical outlining.
    """

    procedure: str
    structured_identity: str
    identity: str
    selected_node_ids: tuple[str, ...]
    private_roots: tuple[str, ...]
    parent: object = field(repr=False, compare=False)
    original_nodes: tuple[object, ...] = field(repr=False, compare=False)
    units: tuple[tuple[object, object, object | None], ...] = field(repr=False, compare=False)

    def validate(self, analysis, procedure, selected):
        graph, identities = analysis._selected_source(procedure, selected)
        if (procedure != self.procedure or graph.identity != self.structured_identity
                or identities != self.selected_node_ids
                or analysis._worksharing_native_completions.get(self.identity) is not self):
            raise CompilationError('native worksharing completion lacks registered original-team authority')
        self.parent.validate(analysis, procedure, self.parent.selected_node_ids)
        return self

    def public(self):
        return {'schema_version': WORKSHARING_NATIVE_COMPLETION_VERSION,
                'proof_role': 'original_team_native_worksharing_completion',
                'proof_identity': self.identity, 'available': True,
                'reason': 'selected original native worksharing units complete before coherence commit',
                'procedure': self.procedure, 'structured_identity': self.structured_identity,
                'parent_completion_identity': self.parent.identity,
                'selected_original_nodes': list(self.selected_node_ids),
                'private_resources': list(self.private_roots), 'worksharing_units': len(self.units),
                'caller_contract': 'all members of the original team at the original reached worksharing units',
                'requires_serial_caller': False, 'has_openmp_in_closure': True,
                'has_opaque_calls_in_closure': False,
                'join': 'original synchronized worksharing completion',
                'retains_original_team_and_directives': True,
                'coherence_protocol': 'coordinator host-begin; team barrier; original work; completion; coordinator host-end; team barrier',
                'standalone_execution_authority': False, 'native_effects_authority': True,
                'gpu_legality_established': False}


def prove_worksharing_native_completion(analysis, procedure, joined, selected):
    """Authenticate one contiguous native subsegment of a complete joined team.

    Individual associated DOs or complete uniform branch constructs may be
    selected. Every selected construct must retain its whole original subtree;
    units cannot be skipped or taken from differently guarded sibling arms.
    Source effects and physical sections still require their separate proofs.
    """
    parent = analysis.joined_completion(procedure, joined)
    units = []
    _joined_completion_facts(analysis, procedure, joined, worksharing=units)
    if not units:
        raise CompilationError('native worksharing subsegment requires a separate original PARALLEL and DO')
    if any(_directive(ending) == 'end do nowait' for _, _, ending in units if ending is not None):
        raise CompilationError('native worksharing subsegments require completion at every original DO; NOWAIT remains native')
    graph, identities = analysis._selected_source(procedure, selected)
    if not identities or len(identities) > analysis.operation_limit:
        raise CompilationError('native worksharing selection requires a bounded nonempty original segment')
    originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
    owned = {id(node) for identity in parent.selected_node_ids
             for original in graph.source_nodes(identity) for node in walk(original)}
    unit_indices = []
    for node in originals:
        if id(node) not in owned:
            raise CompilationError('native worksharing selection must remain inside its original joined team')
        if type(node).__name__ not in {'Block_Nonlabel_Do_Construct', 'If_Construct'}:
            raise CompilationError('native worksharing selection requires associated original DOs or complete uniform branches')
        descendants = {id(item) for item in walk(node)}
        covered = [index for index, (_, loop, _) in enumerate(units) if id(loop) in descendants]
        if not covered or (type(node).__name__ == 'Block_Nonlabel_Do_Construct'
                           and not any(node is units[index][1] for index in covered)):
            raise CompilationError('native worksharing selection must retain complete associated original DOs')
        unit_indices.extend(covered)
    if unit_indices != list(range(unit_indices[0], unit_indices[-1] + 1)):
        raise CompilationError('native worksharing selection must retain a contiguous ordered set of original units')
    selected_units = tuple(units[index] for index in unit_indices)
    record = {'version': WORKSHARING_NATIVE_COMPLETION_VERSION, 'parent': parent.identity,
              'selected': identities, 'units': [graph.node_id(loop) for _, loop, _ in selected_units]}
    identity = sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return WorksharingNativeCompletionProof(procedure, graph.identity, identity, identities,
                                            parent.private_roots, parent, originals, selected_units)


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
    # The enclosing proof authenticated these original loop objects already.
    # Revalidating the complete source graph for every possible sibling makes
    # one requested unit quadratic in the size of a larger original team.
    matches = [(opening, loop, ending) for opening, loop, ending in units
               if (graph.node_id(loop),) == identities]
    if len(matches) != 1:
        raise CompilationError('worksharing selection must be exactly one associated original DO')
    opening, loop, ending = matches[0]
    record = {'version': WORKSHARING_COMPLETION_VERSION, 'parent': parent.identity,
              'selected': identities, 'barrier': 'explicit' if ending is not None else 'implicit'}
    identity = sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return WorksharingCompletionProof(procedure, graph.identity, identity, identities,
                                      parent.private_roots, parent, opening, loop, ending)
