"""Verify source-bound authority for additive full-team scope companions.

This module proves which original calls may enter a collective dispatcher. It
does not insert barriers, choose a policy, or authorize changing other callers.
Runtime allocation/descriptor agreement and bounded collective execution remain
the responsibility of the generated dispatcher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.emission.fortran.formatting import _fortran_list
from compiler.frontend.source_effects import Binding, Routine, SourceEffects, _children, _kind, _part
from compiler.ir import CompilationError


def _span(node):
    item = getattr(node, "item", None)
    span = getattr(item, "fort_original_span", getattr(item, "span", None))
    if span is None:
        raise CompilationError("collective participation requires exact original statement spans")
    return tuple(span)


def _digest_span(path, first, last):
    lines = path.read_text().splitlines(keepends=True)
    if not 1 <= first <= last <= len(lines):
        raise CompilationError("collective participation span is outside its original source")
    return sha256("".join(lines[first - 1:last]).encode()).hexdigest()


def _directive(node):
    if _kind(node) != "Comment":
        return None
    text = str(node).strip().lower()
    return text[5:].strip() if text.startswith("!$omp") else None


def _nodes(routine):
    # fparser puts an opening directive before the first executable statement
    # in Specification_Part. Both parts belong to the same lexical procedure.
    specification = []
    for node in _children(_part(routine.scope.node, "Specification_Part")):
        specification.extend(_children(node) if _kind(node) == "Implicit_Part" else (node,))
    return (*specification, *_children(routine.execution))


def _parallel_clauses(text, host_threads):
    if not re.match(r"parallel(?:\s|$)", text):
        raise CompilationError("collective call requires a plain lexical OpenMP parallel team")
    rest = text[len("parallel"):].strip()
    clauses = {}
    while rest:
        match = re.match(r"([a-z_]+)\s*\(([^()]*)\)\s*(?:,\s*)?", rest)
        if not match:
            raise CompilationError("unsupported lexical parallel directive or clause")
        name, value = match.groups()
        if name not in {"default", "shared", "private", "firstprivate", "num_threads"} or name in clauses:
            raise CompilationError("unsupported or repeated lexical parallel clause: " + name)
        if name == "default":
            if value.strip() not in {"none", "shared"}:
                raise CompilationError("collective parallel default must be shared or none")
            clauses[name] = value.strip()
        elif name == "num_threads":
            if not value.strip().isdigit() or int(value) != host_threads:
                raise CompilationError("lexical parallel thread count must match the fixed host budget")
            clauses[name] = int(value)
        else:
            names = tuple(item.strip() for item in value.split(","))
            if not names or any(not re.fullmatch(r"[a-z][a-z0-9_]*", item) for item in names):
                raise CompilationError("parallel data-sharing clauses require whole names")
            if len(names) != len(set(names)):
                raise CompilationError("parallel data-sharing clause repeats a name")
            clauses[name] = names
        rest = rest[match.end():].strip()
    return clauses


@dataclass(frozen=True)
class CollectiveCallSite:
    """Verified original call and bindings for a compiler-owned source edit."""

    routine: Routine = field(repr=False, compare=False)
    node: object = field(repr=False, compare=False)
    first_line: int
    last_line: int
    team_first_line: int
    team_last_line: int
    span_sha256: str
    team_span_sha256: str
    source_sha256: str
    bindings: dict[str, Binding] = field(repr=False, compare=False)
    shared_array_roots: tuple[str, ...]
    immutable_control_roots: tuple[str, ...]

    @property
    def source(self):
        return self.routine.scope.path

    def call_text(self, companion_name):
        """Render only this call; the owner must provide companion visibility."""
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", companion_name):
            raise CompilationError("collective companion must have a valid Fortran procedure name")
        actuals = [str(value) for value in _children(self.node.items[1])]
        short = f"call {companion_name}({', '.join(actuals)})"
        lines = [short] if len(short) <= 132 else _fortran_list(f"call {companion_name}(", actuals, ")", 0)
        return "\n".join(lines) + "\n"

    def public(self):
        return {
            "source": str(self.source), "source_sha256": self.source_sha256,
            "caller": self.routine.qualified, "first_line": self.first_line,
            "last_line": self.last_line, "span_sha256": self.span_sha256,
            "team_first_line": self.team_first_line, "team_last_line": self.team_last_line,
            "team_span_sha256": self.team_span_sha256, "uniform_guard": "unconditional",
            "resource_mapping": {formal: binding.root for formal, binding in self.bindings.items()},
            "shared_array_roots": list(self.shared_array_roots),
            "immutable_control_roots": list(self.immutable_control_roots),
        }


@dataclass(frozen=True)
class CollectiveParticipation:
    entry: str
    host_threads: int
    sites: tuple[CollectiveCallSite, ...]

    def public(self):
        return {"schema_version": 2, "kind": "omp_full_team", "dispatch": "qualified_companion",
                "entry": self.entry, "expected_omp_level": 1, "host_threads": self.host_threads,
                "call_sites": [site.public() for site in self.sites],
                "other_callers": "unchanged",
                "runtime_requirements": ["team_metadata_before_barriers", "full_team_allocation_agreement",
                                         "exact_shared_descriptors", "one_context_and_ordered_collective_execution"]}


def _resolve_call(analysis, routine, node):
    target, arguments = node.items
    actuals = tuple(_children(arguments))
    if _kind(target) != "Name" or any(_kind(value) != "Name" for value in actuals):
        raise CompilationError("collective calls require direct positional whole-name actuals")
    bindings = [analysis._actual_binding(routine.scope, value) for value in actuals]
    matches = []
    for candidate in analysis._candidates(routine.scope, target):
        callee = analysis.routines.get(candidate)
        if callee is None or len(callee.arguments) != len(bindings):
            continue
        formals = [callee.scope.bindings.get(name) for name in callee.arguments]
        if all(formal is not None and actual is not None and None not in (formal.kind, actual.kind)
               and formal.signature() == actual.signature() for formal, actual in zip(formals, bindings, strict=True)):
            matches.append(callee)
    if len(matches) != 1:
        raise CompilationError("collective call target is unresolved or ambiguous: " + str(target))
    callee = matches[0]
    return callee, {"argument::" + formal: actual
                    for formal, actual in zip(callee.arguments, bindings, strict=True)}


def _effects(analysis, procedure, mapping=None, active=(), budget=None):
    """Map source effects transitively, preserving hidden canonical roots."""
    if procedure in active:
        raise CompilationError("collective source effects cannot be recursive")
    summary = analysis.summarize(procedure)
    if not summary["complete"]:
        raise CompilationError("collective source effect closure is incomplete: " + procedure)
    mapping = mapping or {}
    budget = [analysis.operation_limit] if budget is None else budget
    effects = {}
    for item in summary["operations"]:
        budget[0] -= 1
        if budget[0] < 0:
            raise CompilationError("collective mapped effect budget exhausted")
        kind = item["kind"]
        if kind in {"read", "write", "overwrite", "descriptor_read"}:
            root = mapping.get(item["resource"], item["resource"])
            effects.setdefault(root, set()).add(kind)
        elif kind == "call":
            child_mapping = {formal: mapping.get(actual, actual)
                             for formal, actual in item["resource_mapping"].items()}
            for root, actions in _effects(analysis, item["procedure"], child_mapping,
                                          active + (procedure,), budget).items():
                effects.setdefault(root, set()).update(actions)
        elif kind == "native_contract":
            raise CompilationError("collective native participation requires available source, not opaque effects")
    for root in summary["definition_changes"]:
        effects.setdefault(mapping.get(root, root), set()).add("write")
    return effects


def _threadprivate(analysis):
    roots = set()
    scopes = [*analysis.modules.values(), *(routine.scope for routine in analysis.routines.values())]
    for scope in scopes:
        for node in walk(_part(scope.node, "Specification_Part")):
            directive = _directive(node)
            if directive is None or not directive.startswith("threadprivate"):
                continue
            match = re.fullmatch(r"threadprivate\s*\(([^()]*)\)", directive)
            if not match:
                raise CompilationError("unresolved threadprivate source association")
            for name in match[1].split(","):
                binding = analysis._binding(scope, name.strip())
                if binding is None:
                    raise CompilationError("unresolved threadprivate source association")
                roots.add(binding.root)
    return roots


def _site(analysis, entry, record, captures, host_threads, threadprivate):
    if not isinstance(record, dict) or record.get("uniform_guard") != "unconditional":
        raise CompilationError("collective call-site facts require unconditional participation")
    source = record.get("source")
    if not isinstance(source, str) or source not in analysis.sources:
        raise CompilationError("collective call site is not a supplied original source")
    if "source_sha256" in record and record["source_sha256"] != analysis.sources[source]:
        raise CompilationError("collective call-site source hash differs")
    positions = [record.get(key) for key in ("first_line", "last_line", "team_first_line", "team_last_line")]
    if any(type(value) is not int for value in positions):
        raise CompilationError("collective call-site facts require exact integer line spans")
    first, last, team_first, team_last = positions
    if not team_first < first <= last < team_last:
        raise CompilationError("collective call must lie inside its lexical team")
    path = Path(source)
    digest = _digest_span(path, first, last)
    if record.get("span_sha256") != digest:
        raise CompilationError("collective call-site span hash differs")
    team_digest = _digest_span(path, team_first, team_last)
    if "team_span_sha256" in record and record["team_span_sha256"] != team_digest:
        raise CompilationError("collective team span hash differs")
    candidates = []
    for routine in analysis.routines.values():
        if str(routine.scope.path) != source:
            continue
        for node in walk(routine.execution):
            item = getattr(node, "item", None)
            if (_kind(node) == "Call_Stmt" and item is not None
                    and getattr(item, "fort_original_span", item.span) == (first, last)):
                candidates.append((routine, node))
    if len(candidates) != 1:
        raise CompilationError("collective call span is unavailable or ambiguous")
    routine, node = candidates[0]
    if "caller" in record and record["caller"] != routine.qualified:
        raise CompilationError("collective caller identity differs from the asserted call site")
    nodes = _nodes(routine)
    if not any(value is node for value in nodes):
        raise CompilationError("collective call must be unconditional at the lexical parallel level")
    selected, mapping = _resolve_call(analysis, routine, node)
    if selected.qualified != entry.qualified:
        raise CompilationError("collective call site does not resolve to the requested entry")
    starts = [index for index, value in enumerate(nodes)
              if _directive(value) is not None and _span(value)[0] == team_first]
    ends = [index for index, value in enumerate(nodes)
            if _directive(value) == "end parallel" and _span(value)[1] == team_last]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise CompilationError("collective team span lacks exact lexical parallel boundaries")
    start, stop = starts[0], ends[0]
    clauses = _parallel_clauses(_directive(nodes[start]), host_threads)
    team = nodes[start + 1:stop]
    if not any(value is node for value in team):
        raise CompilationError("collective call is outside its asserted lexical team")
    # Prove this lexical team is not inside another directive region. Unsupported
    # syntax cannot silently act as an opening/closing construct around it.
    depth = 0
    for value in nodes[:start]:
        directive = _directive(value)
        if directive is None:
            continue
        if re.match(r"parallel(?:\s|$)", directive):
            _parallel_clauses(directive, host_threads)
            depth += 1
        elif directive == "end parallel" and depth:
            depth -= 1
        else:
            raise CompilationError("unsupported OpenMP directive before collective team")
    if depth:
        raise CompilationError("nested lexical teams cannot authorize collective dispatch")
    writes = set()
    effect_budget = [analysis.operation_limit]
    for value in team:
        if _kind(value) == "Comment" and _directive(value) is None:
            continue
        if _kind(value) != "Call_Stmt":
            raise CompilationError("collective lexical team permits only unconditional direct calls")
        # Line-based replacement cannot safely replace half a semicolon line.
        if value is not node and not (_span(value)[1] < first or _span(value)[0] > last):
            raise CompilationError("collective call shares an original edit span with another statement")
        callee, bindings = _resolve_call(analysis, routine, value)
        mapped = {formal: binding.root for formal, binding in bindings.items()}
        for root, actions in _effects(analysis, callee.qualified, mapped, budget=effect_budget).items():
            if actions & {"write", "overwrite"}:
                writes.add(root)
    entry_effects = _effects(analysis, entry.qualified)
    # Hidden captures must resolve to the same root in the caller, rather than
    # relying on an identically spelled, potentially shadowed variable.
    for root in set(entry_effects) - set(mapping):
        names = set(routine.scope.bindings) | set(routine.scope.imports)
        for scope in analysis.modules.values():
            names.update(scope.bindings)
            names.update(scope.imports)
        found = [binding for name in names
                 if (binding := analysis._binding(routine.scope, name)) and binding.root == root]
        if not found:
            raise CompilationError("collective hidden capture is not visible at its call site: " + root)
        mapping[root] = found[0]
    shared, private = set(), set()
    for clause in ("shared", "private", "firstprivate"):
        for name in clauses.get(clause, ()):
            binding = analysis._binding(routine.scope, name)
            if binding is None:
                raise CompilationError("parallel data-sharing name is unresolved: " + name)
            (shared if clause == "shared" else private).add(binding.root)
    if shared & private:
        raise CompilationError("parallel data-sharing clauses conflict")
    arrays, controls, seen = set(), set(), {}
    for formal, binding in mapping.items():
        fact = captures.get(formal)
        if (not isinstance(fact, dict) or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False):
            raise CompilationError("collective capture lacks stable whole-root facts: " + formal)
        if binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("collective capture association is uncertain: " + formal)
        if binding.root in private | threadprivate:
            raise CompilationError("collective capture is private, firstprivate, or threadprivate: " + formal)
        if clauses.get("default", "shared") == "none" and binding.root not in shared:
            raise CompilationError("collective capture is not explicitly shared: " + formal)
        prior = seen.setdefault(binding.root, formal)
        if prior != formal and ((entry_effects.get(formal, set()) | entry_effects.get(prior, set()))
                                & {"write", "overwrite"}):
            raise CompilationError("collective writable captures alias: " + binding.root)
        if binding.rank:
            if fact.get("association") != "shared_whole_storage" or fact.get("descriptor_uniform") is not True:
                raise CompilationError("collective arrays require shared whole-storage descriptor facts: " + formal)
            arrays.add(binding.root)
        else:
            if fact.get("association") != "shared_immutable_control" or binding.root in writes:
                raise CompilationError("collective control is not shared and immutable throughout its team: " + formal)
            controls.add(binding.root)
    return CollectiveCallSite(routine, node, first, last, team_first, team_last, digest,
                              team_digest, analysis.sources[source], mapping,
                              tuple(sorted(arrays)), tuple(sorted(controls)))


def verify_collective_participation(analysis: SourceEffects, entry, facts, *, host_threads):
    """Validate schema-2 source authority; return only specifically proved calls."""
    if not isinstance(facts, dict) or facts.get("schema_version") != 2:
        raise CompilationError("collective capture facts require schema_version 2")
    if facts.get("sources") != analysis.sources:
        raise CompilationError("collective capture facts do not match the supplied source hashes")
    if type(host_threads) is not int or not 1 <= host_threads < 2**31:
        raise CompilationError("collective host budget must be a positive default-integer thread count")
    participation = facts.get("participation")
    if (not isinstance(participation, dict) or participation.get("kind") != "omp_full_team"
            or participation.get("dispatch") != "qualified_companion"
            or type(participation.get("expected_omp_level")) is not int
            or participation.get("expected_omp_level") != 1
            or type(participation.get("host_threads")) is not int
            or participation.get("host_threads") != host_threads):
        raise CompilationError("collective participation requires a fixed level-one qualified companion contract")
    names = [name for name in analysis.routines
             if name == str(entry).lower() or ("::" not in str(entry) and name.split("::")[-1] == str(entry).lower())]
    if len(names) != 1 or participation.get("entry") != names[0]:
        raise CompilationError("collective participation entry is unavailable, ambiguous, or mismatched")
    routine = analysis.routines[names[0]]
    if not _children(routine.execution) or any(_kind(node) != "Call_Stmt" for node in _children(routine.execution)
                                             if _kind(node) != "Comment"):
        raise CompilationError("collective companions initially require a call-only owning entry")
    if any(_directive(node) is not None for node in walk(routine.scope.node)):
        raise CompilationError("collective owning entry must not introduce its own OpenMP directives")
    captures, records = facts.get("captures"), participation.get("call_sites")
    if not isinstance(captures, dict) or not isinstance(records, list) or not records or len(records) > 64:
        raise CompilationError("collective participation requires captures and between 1 and 64 call sites")
    threadprivate = _threadprivate(analysis)
    sites = tuple(_site(analysis, routine, record, captures, host_threads, threadprivate) for record in records)
    keys = [(str(site.source), site.first_line, site.last_line) for site in sites]
    if len(keys) != len(set(keys)):
        raise CompilationError("collective participation repeats a call site")
    ordered = sorted(keys)
    if any(a[0] == b[0] and a[2] >= b[1] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise CompilationError("collective call-site edit spans overlap")
    analysis.inputs.verify()
    return CollectiveParticipation(routine.qualified, host_threads, sites)
