"""Original native groups with independently bounded, deferred effect units.

These records establish source selections, not completion or GPU legality.
Only a separately authenticated whole-group native proof may consume a unit's
local graph. Original parser nodes and attached directives are never copied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.ir import CompilationError

NATIVE_GROUP_STRUCTURE_VERSION = 2
NATIVE_GROUP_SCAN_LIMIT = 65536
_CONSTRUCTS = {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def directive(node):
    text = str(node).lstrip().lower()
    return text[5:].strip() if _kind(node) == "Comment" and text.startswith("!$omp") else None


@dataclass
class _Scan:
    visits: int = 0

    def visit(self):
        self.visits += 1
        if self.visits > NATIVE_GROUP_SCAN_LIMIT:
            raise CompilationError("native group original-source scan budget exhausted")


def _roots(nodes):
    """Keep original roots after a bounded, nonrecursive syntax traversal.

    Count actual parser-node visits, including expression payloads. A shared
    prefix can occur both as an exposed node and within its original construct;
    repeated visits count against the same bound. Container iterators avoid an
    eager descendant list or an unbounded temporary child stack.
    """
    scan, traversal, descendants = _Scan(), _Scan(), set()
    for root in nodes:
        stack = [iter((root,))]
        while stack:
            try:
                child = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            traversal.visit()
            if isinstance(child, (tuple, list)):
                stack.append(iter(child))
                continue
            if not (hasattr(child, "content") or hasattr(child, "items")):
                continue
            scan.visit()
            if child is not root:
                descendants.add(id(child))
            stack.append(iter(_children(child)))
    return tuple(node for node in nodes if id(node) not in descendants), scan.visits, traversal.visits


def original_sequence(items, peeled, scan=None):
    """Expose original leading comments without modifying their construct.

    fparser may attach a previous group's ending directive to the next loop.
    Keeping the directive and the loop as separate original objects permits
    selecting the former without absorbing the latter.
    """
    scan = scan or _Scan()
    result = []
    for node in items:
        scan.visit()
        if _kind(node) in _CONSTRUCTS and id(node) not in peeled:
            children = list(_children(node))
            prefix = []
            while children and _kind(children[0]) == "Comment":
                scan.visit()
                prefix.append(children.pop(0))
            peeled[id(node)] = tuple(children)
            result.extend(prefix)
        result.append(node)
    return tuple(result)


def original_section_loops(items, index, *, visit=None, limit=256):
    """Select complete explicit SECTION bodies, retaining their original DOs.

    The caller must expose attached original directives first. This narrow
    syntax contract grants neither completion nor section independence: only
    a whole original joined-team proof may consume the returned selections.
    """
    if directive(items[index]) != "sections":
        raise CompilationError("native SECTIONS requires a plain original SECTIONS directive")
    loops, index = [], index + 1
    while index < len(items):
        if visit is not None:
            visit()
        node, text = items[index], directive(items[index])
        if _kind(node) == "Comment" and text is None:
            index += 1
            continue
        if text in {"end sections", "end sections nowait"}:
            if not loops:
                raise CompilationError("native SECTIONS requires at least one explicit SECTION body")
            return tuple(loops), index + 1
        if text != "section":
            raise CompilationError("native SECTIONS requires explicit SECTION and one complete counted DO body")
        index += 1
        while index < len(items) and _kind(items[index]) == "Comment" and directive(items[index]) is None:
            if visit is not None:
                visit()
            index += 1
        if index >= len(items) or _kind(items[index]) != "Block_Nonlabel_Do_Construct":
            raise CompilationError("native SECTION requires one complete original counted DO body")
        if len(loops) >= limit:
            raise CompilationError("native SECTION source-unit budget exhausted")
        loops.append(items[index])
        index += 1
    raise CompilationError("native SECTIONS requires a matching original END SECTIONS")


@dataclass(frozen=True)
class NativeGroupCandidate:
    original_nodes: tuple = field(compare=False, repr=False)
    items: tuple = field(compare=False, repr=False)
    scanned_source_nodes: int = 0
    source_traversal_entries: int = 0


def grouped_original_sequence(items, peeled):
    """Find complete syntactic group ranges; no token or completion is granted."""
    scan = _Scan()
    nodes = original_sequence(items, peeled, scan)
    result, index = [], 0
    while index < len(nodes):
        scan.visit()
        opening = directive(nodes[index]) or ""
        if not re.match(r"parallel(?:\s|$)", opening):
            result.append(nodes[index])
            index += 1
            continue
        combined = bool(re.match(r"parallel\s+do(?:\s|$)", opening))
        if combined:
            loop = index + 1
            while loop < len(nodes) and _kind(nodes[loop]) == "Comment" and directive(nodes[loop]) is None:
                scan.visit()
                loop += 1
            if loop >= len(nodes) or _kind(nodes[loop]) != "Block_Nonlabel_Do_Construct":
                result.append(nodes[index])
                index += 1
                continue
            end = loop + 1
            while end < len(nodes) and _kind(nodes[end]) == "Comment" and directive(nodes[end]) is None:
                scan.visit()
                end += 1
            if end < len(nodes) and directive(nodes[end]) == "end parallel do":
                end += 1
            elif end < len(nodes) and (directive(nodes[end]) or "").startswith("end parallel"):
                # A mismatched explicit join must not become an implicit one.
                result.append(nodes[index])
                index += 1
                continue
            else:
                end = loop + 1
        else:
            end, nesting = index + 1, 1
            while end < len(nodes):
                scan.visit()
                text = directive(nodes[end]) or ""
                if re.match(r"parallel(?:\s|$)", text):
                    nesting += 1
                elif text == "end parallel":
                    nesting -= 1
                    if not nesting:
                        end += 1
                        break
                end += 1
            if nesting:
                result.append(nodes[index])
                index += 1
                continue
        selected = tuple(nodes[index:end])
        roots, source_visits, traversal_entries = _roots(selected)
        result.append(NativeGroupCandidate(roots, selected, source_visits, traversal_entries))
        index = end
    return tuple(result)


@dataclass(frozen=True)
class NativeGroupUnit:
    id: str
    kind: str
    guard: tuple[str, ...]
    original_nodes: tuple = field(compare=False, repr=False)
    structure: object = field(compare=False, repr=False)
    selected_node_ids: tuple[str, ...] = ()

    def public(self):
        return {"id": self.id, "kind": self.kind, "guard": list(self.guard),
                "structured_identity": self.structure.identity,
                "structured_version": self.structure.version,
                "selected_node_ids": list(self.selected_node_ids),
                "operation_count": self.structure.operation_count,
                "operation_limit": self.structure.operation_limit}


@dataclass(frozen=True)
class NativeGroupStructure:
    node_id: str
    identity: str
    original_nodes: tuple = field(compare=False, repr=False)
    units: tuple[NativeGroupUnit, ...] = ()
    available: bool = False
    reason: str | None = None
    control_count: int = 0
    unit_limit: int = 256
    control_limit: int = 256
    scanned_source_nodes: int = 0
    scanned_control_items: int = 0
    source_traversal_entries: int = 0

    def matches(self, selected):
        # Never map a member AST to its enclosing group. Even an original
        # opening, associated loop or join alone is an incomplete selection.
        if not isinstance(selected, (tuple, list)):
            return False
        return len(selected) == len(self.original_nodes) and all(
            item is original for item, original in zip(selected, self.original_nodes, strict=True))

    def public(self):
        return {"schema_version": NATIVE_GROUP_STRUCTURE_VERSION, "group_id": self.node_id,
                "group_identity": self.identity, "available": self.available, "reason": self.reason,
                "unit_count": len(self.units), "unit_limit": self.unit_limit,
                "control_count": self.control_count, "control_limit": self.control_limit,
                "scanned_source_nodes": self.scanned_source_nodes,
                "source_scan_limit": NATIVE_GROUP_SCAN_LIMIT,
                "source_traversal_entries": self.source_traversal_entries,
                "source_traversal_limit": NATIVE_GROUP_SCAN_LIMIT,
                "scanned_control_items": self.scanned_control_items,
                "control_scan_limit": NATIVE_GROUP_SCAN_LIMIT,
                "units": [unit.public() for unit in self.units],
                "effect_authority": "requires registered complete original native group proof",
                "execution": "unchanged complete original group; no cuts or GPU legality",
                "unknown_effect_policy": "reject whole group; no truncated or inferred effects"}


def build_native_group(analysis, routine, authority_identity, node_id, candidate, guard, peeled, unit_builder):
    """Partition source proof work, retaining all guards and native joins.

    Unit graphs establish bounded syntax only. Calls, lifetime effects and
    semantic completion remain obligations of the native effect consumer.
    """
    units, scan, controls = [], _Scan(), 0
    limit = analysis.operation_limit

    def add(kind, original, guards, header=None):
        if len(units) >= limit:
            raise CompilationError("native group source-unit budget exhausted")
        identity = node_id + "/unit" + str(len(units))
        structure = unit_builder(identity, (original,), tuple(guards), header)
        if not structure.available:
            raise CompilationError("native group unit skeleton unavailable: " + "; ".join(structure.reasons))
        selected = structure.node_id(header, "condition") if header is not None else structure.node_id(original)
        units.append(NativeGroupUnit(identity, kind, tuple(guards), (original,), structure, (selected,)))

    def body(items, guards, depth):
        nonlocal controls
        if depth > analysis.depth_limit:
            raise CompilationError("native group control depth budget exhausted")
        items = original_sequence(items, peeled, scan)
        index = 0
        while index < len(items):
            scan.visit()
            node = items[index]
            kind, text = _kind(node), directive(node)
            if kind == "Comment" and text is None:
                index += 1
                continue
            if kind == "If_Construct":
                previous, branch, header = [], [], None
                for item in peeled.get(id(node), _children(node)):
                    scan.visit()
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        controls += 1
                        if controls > limit:
                            raise CompilationError("native group control-node budget exhausted")
                        if header is not None:
                            branch_guard = (*guards, *(".not.(" + condition + ")" for condition in previous))
                            if _kind(header) != "Else_Stmt":
                                condition = header.items[0]
                                add("condition", condition, branch_guard, header)
                                branch_guard = (*branch_guard, str(condition))
                                previous.append(str(condition))
                            body(branch, branch_guard, depth + 1)
                        header, branch = item, []
                    else:
                        branch.append(item)
                index += 1
                continue
            if text == "sections":
                loops, index = original_section_loops(items, index, visit=scan.visit, limit=limit)
                for loop in loops:
                    add("section", loop, guards)
                continue
            if text is None or not re.match(r"do(?:\s|$)", text):
                raise CompilationError("deferred native group requires complete worksharing DO units: " + (text or kind))
            index += 1
            while index < len(items) and _kind(items[index]) == "Comment" and directive(items[index]) is None:
                scan.visit()
                index += 1
            if index >= len(items) or _kind(items[index]) != "Block_Nonlabel_Do_Construct":
                raise CompilationError("deferred native DO lacks its complete original associated loop")
            add("worksharing", items[index], guards)
            index += 1
            while index < len(items) and _kind(items[index]) == "Comment" and directive(items[index]) is None:
                scan.visit()
                index += 1
            if index < len(items) and directive(items[index]) in {"end do", "end do nowait"}:
                index += 1
            elif index < len(items) and (directive(items[index]) or "").startswith("end do"):
                raise CompilationError("deferred native DO has an unsupported ending directive")

    reason = None
    try:
        items = candidate.items
        opening = directive(items[0]) or ""
        if re.match(r"parallel\s+do(?:\s|$)", opening):
            loops = [item for item in items[1:] if _kind(item) != "Comment"]
            if len(loops) != 1 or _kind(loops[0]) != "Block_Nonlabel_Do_Construct":
                raise CompilationError("deferred combined native group lacks one complete original DO")
            add("worksharing", loops[0], guard)
        else:
            body(items[1:-1], tuple(guard), 0)
    except CompilationError as error:
        # Partial materialization cannot grant any authority or truncate the
        # original operation. Its unchanged source remains a boundary.
        reason = str(error)
        units = []
    payload = {"version": NATIVE_GROUP_STRUCTURE_VERSION, "procedure": routine.qualified,
               "authority_identity": authority_identity, "node_id": node_id,
               "source": [str(node) for node in candidate.original_nodes], "guard": list(guard),
               "units": [unit.public() for unit in units], "reason": reason,
               "control_count": controls, "operation_limit": limit,
               "scanned_source_nodes": candidate.scanned_source_nodes,
               "source_traversal_entries": candidate.source_traversal_entries,
               "scanned_control_items": scan.visits, "scan_limit": NATIVE_GROUP_SCAN_LIMIT}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return NativeGroupStructure(node_id, identity, candidate.original_nodes, tuple(units),
                                reason is None, reason, controls, limit, limit,
                                candidate.scanned_source_nodes, scan.visits,
                                candidate.source_traversal_entries)
