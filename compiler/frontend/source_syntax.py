"""Original-syntax authority without recursively formatting Fortran trees.

Mutable parser inputs are checked afresh at each demand. Active compilation
sessions seal their originals and then authenticate node identity directly.
Namespace, binding, configuration and source-file checks remain with callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat
from operator import and_, attrgetter, eq, is_

from fparser.common.readfortran import FortranReaderBase
from fparser.two.utils import Base, BlockBase

from compiler.frontend.sealed_syntax import FrozenList
from compiler.ir import CompilationError


@dataclass(frozen=True, slots=True)
class _Sequence:
    original: list
    values: tuple

    def unchanged(self):
        return (type(self.original) is list and len(self.original) == len(self.values)
                and all(map(is_, self.original, self.values)))


@dataclass(frozen=True, slots=True)
class _Node:
    original: Base
    kind: type
    keys: tuple
    values: tuple
    children: tuple
    sequences: tuple
    item: object
    item_kind: type
    item_attributes: tuple


_ITEM_FIELDS = ("span", "fort_original_span", "label", "name", "line", "comment",
                "is_f2py_directive")
_ITEM_FIELD_NAMES = frozenset(_ITEM_FIELDS)


@dataclass(frozen=True, slots=True)
class _Group:
    objects: tuple
    kinds: tuple
    keys: frozenset
    attributes: tuple
    item: bool = False

    def unchanged(self):
        if not all(map(is_, map(type, self.objects), self.kinds)):
            return False
        keys = (vars(value).keys() for value in self.objects)
        if self.item:
            keys = map(and_, keys, repeat(_ITEM_FIELD_NAMES))
        if not all(map(eq, keys, repeat(self.keys))):
            return False
        return all(all(map(is_, map(attrgetter(name), self.objects), original))
                   for name, original in self.attributes)


@dataclass(frozen=True, slots=True)
class _Layout:
    identities: tuple
    groups: tuple
    sequences: tuple

    def unchanged(self):
        return (all(group.unchanged() for group in self.groups)
                and all(sequence.unchanged() for sequence in self.sequences))


class SourceSyntaxSnapshot:
    """Bounded immutable records of original parser objects and syntax values.

    Reader internals and parent links do not specify syntax. Block reader
    identity is retained, but parser caches and file position are not traversed.
    Statement labels, construct names and source spans are syntax/provenance.
    """

    def __init__(self, roots, *, limit=4_194_304):
        self._roots = tuple(roots)
        self._sealed = False
        self._nodes = {}
        self._layouts = {}
        self._active = set()
        self._remaining = limit
        try:
            for root in self._roots:
                if root is not None:
                    self._capture_node(root)
        except RecursionError as error:
            raise CompilationError("original source syntax nesting budget exhausted") from error
        self._layout_limit = 2 * len(self._nodes)
        self._layout_weight = 0
        del self._active, self._remaining

    def _tick(self):
        self._remaining -= 1
        if self._remaining < 0:
            raise CompilationError("original source syntax snapshot budget exhausted")

    def _capture_value(self, value, children, sequences, active):
        self._tick()
        kind = type(value)
        if isinstance(value, Base):
            self._capture_node(value)
            children.append(value)
            return
        if kind in (str, int, bool, bytes, type(None)):
            return
        if kind in (tuple, list, FrozenList):
            if id(value) in active:
                raise CompilationError("original source syntax contains a cycle")
            active.add(id(value))
            if kind is list:
                values = tuple(value)
                sequences.append(_Sequence(value, values))
            for item in value:
                self._capture_value(item, children, sequences, active)
            active.remove(id(value))
            # Immutable tuple identity fixes all of its edges and scalar
            # values. Nested mutable lists have separate shallow records;
            # referenced parser nodes are checked through children below.
            return
        raise CompilationError("original source syntax contains unsupported parser state")

    def _capture_node(self, node):
        if not isinstance(node, Base):
            raise CompilationError("original source syntax requires parser nodes")
        key = id(node)
        if key in self._active:
            raise CompilationError("original source syntax contains a cycle")
        if key in self._nodes:
            return
        self._tick()
        self._active.add(key)
        children, sequences = [], []
        for name, value in vars(node).items():
            if name in {"parent", "item"}:
                continue
            if name == "string" and isinstance(node, BlockBase) and isinstance(value, FortranReaderBase):
                continue
            else:
                self._capture_value(value, children, sequences, set())
        item = getattr(node, "item", None)
        item_attributes = tuple((name, getattr(item, name))
                                for name in _ITEM_FIELDS if hasattr(item, name))
        for _name, value in item_attributes:
            self._capture_value(value, children, sequences, set())
        keys, values = tuple(vars(node)), tuple(vars(node).values())
        self._nodes[key] = _Node(node, type(node), keys, values,
                                 tuple(children), tuple(sequences), item, type(item),
                                 item_attributes)
        self._active.remove(key)

    def checkpoint(self):
        return _SyntaxCheckpoint(self)

    @property
    def immutable(self):
        """All authenticated original syntax and source spans reject writes."""
        return self._sealed

    def seal(self):
        """Return immutable syntax authority for an active compiler session.

        Namespace and configured-file authentication remain separate. Original
        parser attributes and child lists reject writes; copying a node creates
        a mutable projection which cannot pass this snapshot's identity check.
        """
        if self._sealed:
            return self
        checkpoint = self.checkpoint()
        if not all(checkpoint.unchanged(root) for root in self._roots):
            raise CompilationError("cannot seal changed original source syntax")
        from compiler.frontend.sealed_syntax import seal

        seal(self._roots)
        result = SourceSyntaxSnapshot(self._roots)
        result._sealed = True
        return result

    def _layout(self, node):
        """Group immutable baseline records; never cache a validation result."""
        key = id(node)
        if key in self._layouts:
            return self._layouts[key]
        pending, seen, states = [node], set(), []
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            state = self._nodes[identity]
            states.append(state)
            pending.extend(state.children)
        nodes, items, sequences = {}, {}, {}
        for state in states:
            nodes.setdefault(state.keys, []).append(state)
            if state.item is not None:
                items.setdefault(tuple(name for name, _ in state.item_attributes), []).append(state)
            for sequence in state.sequences:
                sequences[id(sequence.original)] = sequence
        groups = []
        for keys, records in nodes.items():
            groups.append(_Group(tuple(state.original for state in records),
                                 tuple(state.kind for state in records), frozenset(keys),
                                 tuple((name, tuple(state.values[index] for state in records))
                                       for index, name in enumerate(keys) if name != "parent")))
        for keys, records in items.items():
            groups.append(_Group(tuple(state.item for state in records),
                                 tuple(state.item_kind for state in records), frozenset(keys),
                                 tuple((name, tuple(state.item_attributes[index][1] for state in records))
                                       for index, name in enumerate(keys)), item=True))
        layout = _Layout(tuple(seen), tuple(groups), tuple(sequences.values()))
        # Repeated subtree requests may overlap. Bound retained baseline
        # indexes independently of the per-demand successful comparisons.
        while self._layouts and self._layout_weight + len(seen) > self._layout_limit:
            oldest = self._layouts.pop(next(iter(self._layouts)))
            self._layout_weight -= len(oldest.identities)
        self._layouts[key] = layout
        self._layout_weight += len(seen)
        return layout


class _SyntaxCheckpoint:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self._unchanged = set()
        self._changed = set()

    def unchanged(self, node):
        """Check original identity and, when mutable, its current descendants."""
        if node is None:
            # An original empty procedure has no Execution_Part. Callers
            # separately check that this is still its registered execution.
            return True
        key = id(node)
        if key in self._unchanged:
            return True
        if key in self._changed:
            return False
        state = self._snapshot._nodes.get(key)
        if state is None or state.original is not node or type(node) is not state.kind:
            return False
        if self._snapshot._sealed:
            return True
        layout = self._snapshot._layout(node)
        unchanged = layout.unchanged()
        if unchanged:
            self._unchanged.update(layout.identities)
        else:
            self._changed.add(key)
        return unchanged
