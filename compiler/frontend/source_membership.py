"""Exact immutable parser membership and subtree overlap coordinates.

The traversal matches fparser's walk, including its tuple handling. Coordinates
are useful only for a tree: shared parser nodes retain membership but disable
the interval shortcut. Callers keep their original walk for that case.
"""

from dataclasses import dataclass
from types import MappingProxyType

from fparser.two.utils import Base


@dataclass(frozen=True, slots=True)
class ExecutionMembership:
    nodes: object
    intervals: object

    @classmethod
    def create(cls, execution):
        nodes, intervals, seen = {}, {}, set()
        pending, position, shared = [("walk", execution)], 0, False
        while pending:
            role, value = pending.pop()
            if role == "walk":
                sequence = value if isinstance(value, (tuple, list)) else (value,)
                pending.extend(("child", child) for child in reversed(sequence))
                continue
            if role == "leave":
                node, start = value
                intervals[id(node)] = (node, start, position)
                continue
            nodes[id(value)] = value
            if isinstance(value, Base):
                identity = id(value)
                shared |= identity in seen
                seen.add(identity)
                start = position
                if hasattr(value, "items") or hasattr(value, "content"):
                    position += 1
                pending.append(("leave", (value, start)))
                pending.append(("walk", value.children))
            elif isinstance(value, tuple):
                pending.extend(("walk", child) for child in reversed(value))
        return cls(MappingProxyType(nodes), None if shared else MappingProxyType(intervals))

    def overlaps(self, selections):
        """Return None when exact source coordinates cannot replace a walk."""
        if self.intervals is None:
            return None
        ranges = []
        for owner, selection in enumerate(selections):
            for node in selection:
                entry = self.intervals.get(id(node))
                if entry is None or entry[0] is not node:
                    return None
                _, start, end = entry
                if start != end:
                    ranges.append((start, end, owner))
        maximum, previous = -1, None
        for start, end, owner in sorted(ranges):
            if start < maximum and previous != owner:
                return True
            if end > maximum:
                maximum, previous = end, owner
        return False
