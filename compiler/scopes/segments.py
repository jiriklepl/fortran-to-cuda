"""Bounded structured owners, with queries evaluated at reached source points.

This is source scheduling, not numerical lowering. Inline source operations stay
native and retain their original OpenMP team. Numerical calls reuse the existing
mode-bearing workers and public planning interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, replace

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.component_bindings import references
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, MODEL_INQUIRIES


@dataclass
class Native:
    nodes: tuple
    effects: dict
    overwrites: set
    bindings: dict
    summary: dict
    kind: str = "native source"
    sections: object = None
    span: tuple = ()
    private_bindings: dict = field(default_factory=dict)
    host_metadata: dict = field(default_factory=dict)
    preserve_original: bool = False
    host_only_reads: dict = field(default_factory=dict)
    host_only_proof: dict = field(default_factory=dict)

    @property
    def coherence_effects(self):
        """Managed obligations only; ``effects`` still reports original reads."""
        return {root: actions for root, actions in self.effects.items() if root not in self.host_only_reads}

    @property
    def coherence_sections(self):
        if self.sections is None or not self.host_only_reads:
            return self.sections
        return replace(self.sections, resources=tuple(section for section in self.sections.resources
                                                     if section.resource not in self.host_only_reads))


def _native_analysis(analysis):
    """Isolate reached native proofs without changing GPU lifetime authority.

    Original source/AST objects remain shared and authenticated. Cache and token
    dictionaries are private, including eviction and persistent-cache counters.
    No caller capture fact or initialized coverage is created by this projection.
    """
    from compiler.frontend.summary_cache import SummaryCache

    result = copy.copy(analysis)
    for name in ("summaries", "_closures", "_native_sections", "_structures", "_segments",
                 "_descriptor_proofs", "_joined_completions", "_numerical_completions",
                 "_worksharing_completions", "_worksharing_native_completions", "_reduction_proofs",
                 "_reduction_candidates", "_omp_reduction_proofs", "_source_scopes", "_associate_scopes",
                 "_source_provenance", "_allocation_authorizations"):
        setattr(result, name, dict(getattr(analysis, name)))
    result._summary_cache = SummaryCache(max_entries=0)
    return result


def _host_read_candidate(builder, binding):
    """A missing GPU capture proof can admit only original module storage."""
    if binding.root in builder.facts.get("captures", {}):
        return False
    parts = binding.root.split("::")
    module = builder.analysis.modules.get(parts[0]) if len(parts) == 2 else None
    forbidden = {"pointer", "optional", "volatile", "asynchronous", "value", "parameter"}
    candidate = (module is not None and module.bindings.get(parts[1]) is binding and binding.rank > 0
            and "allocatable" in binding.attributes and not binding.attributes & forbidden
            and binding.dtype in {"real", "integer"} and binding.kind in {4, 8})
    if not candidate:
        return False
    specification = next((node for node in getattr(module.node, "content", ())
                          if _kind(node) == "Specification_Part"), None)
    for node in walk(specification):
        if _kind(node) in {"Common_Stmt", "Equivalence_Stmt"}:
            return False
        value = directive(node)
        if value is not None and value.startswith("threadprivate"):
            match = re.fullmatch(r"threadprivate\s*\(([^()]*)\)", value)
            if match is None or binding.name.lower() in {name.strip().lower() for name in match[1].split(",")}:
                return False
    return True


@dataclass
class Segment:
    calls: list
    identity: int = -1
    query: tuple = ()
    payload: tuple = ()


@dataclass
class Branch:
    # A None condition denotes ELSE. Conditions are separate native reads so
    # later ELSEIF expressions are never evaluated before preceding guards.
    alternatives: list
    span: tuple = ()


@dataclass
class Association:
    body: list
    span: tuple
    proof: dict


@dataclass
class Delegated:
    call: object
    coordinator: object


def fortran_lines(lines):
    """Wrap generated free-form syntax, including continued OpenMP comments."""
    result = []
    for original in lines:
        line = original
        openmp = line.lstrip().lower().startswith("!$omp")
        if line.lstrip().startswith("!") and not openmp:
            result.append(line)
            continue
        while len(line) > 112:
            quote, boundary = None, None
            for index, character in enumerate(line[:112]):
                if character in {"'", '"'}:
                    quote = None if quote == character else character if quote is None else quote
                elif quote is None and character in {" ", ","} and index > 5:
                    boundary = index + (character == ",")
            if boundary is None:
                raise CompilationError("structured source has an overlong indivisible token")
            result.append(line[:boundary].rstrip() + " &")
            line = ("!$omp& " if openmp else "  & ") + line[boundary:].lstrip()
        result.append(line)
    return result


def directive(node):
    text = str(node).lstrip().lower()
    return text[5:].strip() if _kind(node) == "Comment" and text.startswith("!$omp") else None


def original_roots(nodes):
    """Keep exact recovered constructs once, including attached directives."""
    descendants = {id(child) for node in nodes for child in walk(node) if child is not node}
    unique = {id(node): node for node in nodes}
    return tuple(node for identity, node in unique.items() if identity not in descendants)


def joined_group_completion(builder, nodes):
    """Compatibility view of the registered original-source completion proof.

    Source extraction may carry compiler-owned projections. Only its exact
    registry can recover originals; manufactured spans or attributes grant no
    authority. Simple analysis callers must supply the actual original nodes.
    """
    originals = original_roots(builder.inline.original_selection(nodes) if hasattr(builder, "inline") else tuple(nodes))
    return builder.analysis.joined_completion(builder.entry.qualified, originals).public()


def grouped_nodes(nodes):
    """Keep whole joined groups as one native scheduling operation."""
    expanded = []
    for node in nodes:
        if _kind(node) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}:
            prefix = []
            for item in node.content:
                if _kind(item) != "Comment":
                    break
                prefix.append(item)
            if prefix:
                # fparser attaches directives before the DO header to its
                # construct. Their source positions precede the loop, so expose
                # them to the group parser without mutating the original AST.
                clone = copy.copy(node)
                clone.content = list(node.content[len(prefix):])
                expanded += [*prefix, clone]
                continue
        expanded.append(node)
    nodes = expanded
    result, index = [], 0
    while index < len(nodes):
        start = directive(nodes[index])
        if start is not None and re.match(r"parallel(?:\s|$)", start):
            combined = bool(re.match(r"parallel\s+do(?:\s|$)", start))
            ending = "end parallel do" if combined else "end parallel"
            if combined:
                loop = index + 1
                while loop < len(nodes) and _kind(nodes[loop]) == "Comment" and directive(nodes[loop]) is None:
                    loop += 1
                if loop >= len(nodes) or _kind(nodes[loop]) != "Block_Nonlabel_Do_Construct":
                    raise CompilationError("combined PARALLEL DO requires its original associated loop")
                end = loop + 1
                while end < len(nodes) and _kind(nodes[end]) == "Comment" and directive(nodes[end]) is None:
                    end += 1
                if end < len(nodes) and directive(nodes[end]) == ending:
                    result.append(tuple(nodes[index:end + 1]))
                    index = end + 1
                else:
                    if end < len(nodes) and (directive(nodes[end]) or "").startswith("end parallel"):
                        raise CompilationError("combined PARALLEL DO has a mismatched original join")
                    # Fortran's END PARALLEL DO is optional. The complete DO
                    # itself supplies the join; never absorb following work.
                    result.append(tuple(nodes[index:loop + 1]))
                    index = loop + 1
                continue
            end = index + 1
            while end < len(nodes) and directive(nodes[end]) != ending:
                end += 1
            if end == len(nodes):
                raise CompilationError("native PARALLEL group has no proven join")
            result.append(tuple(nodes[index:end + 1]))
            index = end + 1
        else:
            result.append(nodes[index])
            index += 1
    return result


def statement_span(node):
    items = [item for item in walk(node) if getattr(item, "item", None)]
    if not items:
        raise CompilationError("structured source operation lacks an exact edit span")
    spans = [getattr(item.item, "fort_original_span", item.item.span) for item in items]
    if any(span is None for span in spans):
        raise CompilationError("structured source operation lacks an original edit span")
    return min(span[0] for span in spans), max(span[1] for span in spans)


def fragment(builder, nodes, *, kind="native source", selected=None, private_roots=(), native_metadata=False,
             native_host_reads=False,
             completion=None, span=None):
    """Demand effects from exact original nodes, excluding owner entry events."""
    original = original_roots(builder.inline.original_selection(nodes)) if selected is None else ()
    full_original = original
    analysis = _native_analysis(builder.analysis) if native_host_reads else copy.copy(builder.analysis)
    analysis._native_metadata = native_metadata
    joined = kind == "joined native OpenMP" or (
        kind == "guarded original numerical fallback"
        and any(directive(item) is not None for node in original for item in walk(node)))
    # Bounds and tile guards must retain the completion proof of the exact
    # original team. The numerical outlining token alone cannot authorize
    # native effects; issue an independent native joined-group proof here.
    completion_proof = completion
    if completion_proof is not None:
        from compiler.frontend.worksharing_completion import WorksharingNativeCompletionProof
        if not isinstance(completion_proof, WorksharingNativeCompletionProof):
            raise CompilationError("native subsegment requires a registered native worksharing proof")
        completion_proof.validate(builder.analysis, builder.entry.qualified, original)
        # This validated original token supplies private-state facts only.
        # Issue the copied analysis's authority once, after capture facts have
        # been established below, rather than proving the same parent twice.
    elif joined:
        completion_proof = analysis.joined_completion(builder.entry.qualified, original)
    if completion_proof is not None:
        private_roots = set(private_roots) | set(completion_proof.private_roots)
    original = tuple(node for node in original if _kind(node) != "Comment")
    deferred = analysis.structure(builder.entry.qualified).native_group_for_selection(full_original) if joined else None
    selection = (full_original if deferred is not None else original) if selected is None else selected
    # Validated capture facts borrow only the reached local descriptor. They
    # neither alter original source authority nor authorize allocation changes.
    analysis._segments = {}
    bindings, private_bindings = {}, {}
    examined = original if selected is None else nodes
    descriptor_references = set()
    for node in examined:
        for expression in walk(node):
            if _kind(expression) != "Intrinsic_Function_Reference":
                continue
            name = str(expression.items[0]).lower()
            scope = analysis.source_scope_for(expression, builder.entry.scope)
            arguments = getattr(expression.items[1], "items", ())
            if (name in ARRAY_INQUIRIES | MODEL_INQUIRIES | {"allocated", "present"}
                    and not analysis._binding(scope, name) and not analysis._candidates(scope, name)
                    and not analysis._unknown_exports(scope) and arguments):
                argument = arguments[0]
                if _kind(argument) == "Actual_Arg_Spec":
                    argument = argument.items[1]
                if _kind(argument) == "Name":
                    descriptor_references.add(id(argument))
    referenced = [item for node in examined for item in references(analysis, builder.entry.scope, node)]
    host_metadata = {binding.native_metadata_object.root: binding.native_metadata_object
                     for binding, _ in referenced if hasattr(binding, 'native_metadata_object')}
    payload_roots = {binding.root for binding, reference in referenced if id(reference) not in descriptor_references}
    host_only_reads = {}
    for binding, _ in referenced:
        if hasattr(binding, 'native_metadata_object') or binding.root in host_metadata:
            continue
        if binding.root in private_roots:
            private_bindings[binding.root] = binding
            continue
        bindings[binding.root] = binding
        if native_host_reads and _host_read_candidate(builder, binding):
            # This provisional *native-only* allowance is validated against the
            # complete reached operation below. It grants neither GPU access
            # nor any initialized/defined coverage, and never escapes this fork.
            host_only_reads[binding.root] = binding
            analysis._stable_module_allocatables = analysis._stable_module_allocatables | {binding.root}
            continue
        if binding.rank and "allocatable" in binding.attributes and binding.root in payload_roots:
            builder.capture(binding)
            analysis._stable_module_allocatables = analysis._stable_module_allocatables | {binding.root}
    if completion_proof is not None:
        # Verified local capture facts participate in graph authority. Issue
        # the token against the same reached proof used for materialization.
        if completion is None:
            completion_proof = analysis.joined_completion(builder.entry.qualified, full_original)
        else:
            parent_graph = analysis.structure(builder.entry.qualified)
            completion_proof = analysis.worksharing_native_completion(
                builder.entry.qualified,
                tuple(node for identity in completion.parent.selected_node_ids
                      for node in parent_graph.source_nodes(identity)), full_original)
    if deferred is not None:
        from compiler.scopes.native_atomic import summarize
        summary = summarize(analysis, builder.entry.qualified, selection, completion_proof)
    else:
        summary = analysis.segment_summary(builder.entry.qualified, selection, capture_locals=True)
    if (deferred is None and not summary["complete"] and completion_proof is not None
            and builder.config.scope_execution == "reached"
            and any('budget exhausted' in reason for reason in summary['reasons'])):
        from compiler.scopes.native_atomic import summarize
        summary = summarize(analysis, builder.entry.qualified, selection, completion_proof)
    if not summary["complete"]:
        raise CompilationError("structured native effects incomplete: " + "; ".join(summary["reasons"]))
    if any(operation["kind"] == "boundary" or (operation["kind"] == "control"
           and not (kind == "condition read" and operation["source"].upper() == "CONTINUE"))
           for operation in summary["operations"]):
        raise CompilationError("unsupported exit or lifetime boundary inside structured ownership")
    completion = summary["native_completion"]
    if completion_proof is not None:
        completion = summary["native_completion"] = completion_proof.public()
    if not completion["available"]:
        raise CompilationError("structured native completion unproven: " + completion["reason"])
    effects = {}
    for operation in summary["operations"]:
        if operation["kind"] in {"call", "native_contract"}:
            raise CompilationError("inline native calls require a separately mapped source operation")
        if operation["kind"] in {"read", "write", "overwrite"} and operation["rank"]:
            if operation["resource"] in private_roots:
                continue
            effects.setdefault(operation["resource"], set()).add(
                "read" if operation["kind"] == "read" else "write")
    for root in host_only_reads:
        if any(operation.get("resource") == root and operation["kind"] not in {"read", "descriptor_read"}
               for operation in summary["operations"]):
            raise CompilationError("host-only native storage requires read-only complete effects: " + root)
        if "write" in effects.get(root, ()) or root in summary["guaranteed_whole_overwrites"]:
            raise CompilationError("host-only native storage cannot acquire a managed write: " + root)
    host_only_proof = {}
    if host_only_reads:
        host_only_proof = {"schema_version": 1, "authority": "complete original reached native operation",
            "analysis_identity": summary["analysis_identity"], "summary_identity": summary["summary_identity"],
            "structured_identity": summary.get("structured_identity"), "demand_identity": summary.get("demand_identity"),
            "resources": sorted(host_only_reads), "allocation_changes": False, "calls_or_escapes": False,
            "device_capture": False, "managed_definitions": False,
            "lifetime": "original stable native operation; allocation changes close ownership",
            "range_validation": "allocated guard then undefined BYTE reservation; managed aliases close before native work"}
        host_only_proof["identity"] = hashlib.sha256(json.dumps(host_only_proof, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        host_metadata.update(host_only_reads)
    span = span if span is not None else ((statement_span(nodes[0])[0], statement_span(nodes[-1])[1])
                                        if kind != "condition read" else ())
    sections = analysis.native_sections_for_nodes(builder.entry.qualified, selection, capture_locals=True,
                                                  completion=completion_proof)
    sections = replace(sections, resources=tuple(item for item in sections.resources
                                                if item.resource not in private_roots))
    return Native(tuple(nodes), effects, set(summary["guaranteed_whole_overwrites"]) - set(private_roots),
                  bindings, summary, kind, sections, span, private_bindings, host_metadata,
                  host_only_reads=host_only_reads, host_only_proof=host_only_proof)


def condition_fragment(builder, header, *, native_host_reads=False):
    # The graph authenticates the original header and expression separately;
    # no synthetic assignment may establish source authority.
    condition = header.items[0]
    identity = builder.analysis.structure(builder.entry.qualified).node_id(header, role="condition")
    result = fragment(builder, (condition,), kind="condition read", selected=(identity,), native_host_reads=native_host_reads)
    result.nodes = (condition,)
    result.span = statement_span(header)
    return result


def renamed(builder, node, parameters, *, lexical_scope=None, native_metadata=False):
    """Rewrite resolved variable AST nodes, never string substrings."""
    if native_metadata:
        builder = copy.copy(builder)
        builder.analysis = copy.copy(builder.analysis)
        builder.analysis._native_metadata = True
    def rename_comment(value):
        def clause(match):
            values = []
            for name in match.group(2).split(","):
                binding = builder.analysis._binding(builder.entry.scope, name.strip())
                values.append(parameters.get(binding.root, name.strip()) if binding else name.strip())
            return match.group(1) + "(" + ",".join(values) + ")"
        # Directives are fparser comment nodes. Only parsed admitted variable
        # lists are rewritten; arbitrary comments and literals are untouched.
        return re.sub(r"(private|shared)\s*\(([^()]*)\)", clause, str(value), flags=re.IGNORECASE)

    def clone_ast(value):
        if isinstance(value, (tuple, list)):
            return type(value)(clone_ast(child) for child in value)
        if not isinstance(value, Base):
            return value
        scope = builder.analysis.source_scope_for(value, lexical_scope or builder.entry.scope)
        if _kind(value) == "Actual_Arg_Spec":
            result = copy.copy(value)
            result.items = (value.items[0], clone_ast(value.items[1]))
            return result
        if _kind(value) == "Data_Ref":
            from compiler.frontend.component_bindings import component_access
            access = component_access(builder.analysis, scope, value)
            if hasattr(access.binding, 'native_metadata_object'):
                # Only original lexical native blocks admit these objects.
                # Their fields and coordinates retain their original names;
                # managed input storage has already been made host-current.
                return value
            replacement = parameters.get(access.binding.root)
            if replacement is None:
                replacement = builder.visible(builder.entry, access.binding.root)
            if access.indices:
                replacement += "(" + ",".join(str(clone_ast(index)) for index in access.indices) + ")"
                return F.Data_Ref(replacement) if "%" in replacement else F.Part_Ref(replacement)
            return F.Data_Ref(replacement) if "%" in replacement else F.Name(replacement)
        if _kind(value) == "Name":
            binding = builder.analysis._binding(scope, value)
            if binding is not None and binding.root in parameters:
                replacement = parameters[binding.root]
                return F.Data_Ref(replacement) if "%" in replacement else F.Name(replacement)
            if binding is not None and hasattr(binding, "associate_selector"):
                visible = builder.visible(builder.entry, binding.root)
                return F.Data_Ref(visible) if "%" in visible else F.Name(visible)
            return value
        if (_kind(value) == "Intrinsic_Function_Reference" and str(value.items[0]).lower() == "allocated"):
            arguments = getattr(value.items[1], "items", ())
            if len(arguments) == 1 and _kind(arguments[0]) == "Name":
                binding = builder.analysis._binding(scope, arguments[0])
                if binding and binding.root in parameters and "allocatable" in binding.attributes:
                    # This helper can only be associated after the original
                    # allocation guard. Unallocated invocations execute the
                    # untouched source outside it, with their original guards.
                    builder.capture(binding)
                    return F.Logical_Literal_Constant(".TRUE.")
        if _kind(value) == "Comment":
            result = object.__new__(type(value))
            result.__dict__ = dict(value.__dict__)
            result.items = [rename_comment(value) if directive(value) is not None else str(value)]
            return result
        result = copy.copy(value)
        for attribute in ("content", "items"):
            if hasattr(value, attribute):
                setattr(result, attribute, clone_ast(getattr(value, attribute)))
        return result
    # Readers/items contain open source handles; only typed syntax needs a
    # private copy. Original provenance references remain read-only.
    return str(clone_ast(node))


class StructuredScope:
    def __init__(self, builder, nodes, *, condition_only=False, native_metadata=False, lexical_native=False,
                 native_host_reads=False):
        self.builder, self.nodes = builder, tuple(nodes)
        self.native_metadata = native_metadata
        self.native_host_reads = native_host_reads
        self.calls, self.native, self.segments = [], [], []
        self.guarded = {}
        self.operation_count = 0
        if condition_only:
            if len(nodes) != 1 or _kind(nodes[0]) not in {"If_Then_Stmt", "If_Stmt"}:
                raise CompilationError("reached condition publication requires one original IF header")
            condition = condition_fragment(builder, nodes[0], native_host_reads=native_host_reads)
            # The original header evaluates its expression once, after this
            # publication. This native operation only establishes host reads.
            operation = replace(condition, nodes=())
            self.native.append(operation)
            self.operation_count = len(operation.summary["operations"])
            self.tree = [operation]
        else:
            self.tree = self.parse(nodes)
        if len(self.calls) > 32 or self.operation_count > 256:
            raise CompilationError("bounded structured owner exceeds call/operation budget")
        self.first, self.last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
        original = builder.entry.scope.path.read_text().splitlines()[self.first - 1:self.last]
        if any(line.lstrip().startswith("#") for line in original):
            # Only the original lexical owner can surround a complete native
            # group without reproducing its preprocessing side effects. Its
            # mapped outer directives bound the edit; included statements are
            # analyzed through configured provenance, never edited by prepared
            # line numbers. Numerical outlining and ordinary workers stay strict.
            if not (lexical_native and str(builder.entry.scope.path) in builder.analysis.inputs.entries
                    and len(self.tree) == 1 and isinstance(self.tree[0], Native)
                    and self.tree[0].kind == "joined native OpenMP"):
                raise CompilationError("preprocessor control boundary inside structured ownership")
            operation = self.tree[0]
            operation.preserve_original = True
            # The hooks surround one unchanged operation. Keep their storage
            # obligations conservative; section temporaries must not outlive a
            # preparation block or assume definitions inside an inactive branch.
            operation.sections = replace(operation.sections, available=False, resources=(),
                reason="original configured native group uses whole managed-resource effects")

    def parse(self, nodes, depth=0):
        if depth > 8:
            raise CompilationError("bounded structured owner exceeds branch depth")
        result, run = [], []

        def flush():
            if run:
                result.append(Segment(list(run)))
                run.clear()

        for node in self.builder.inline.grouped_nodes(nodes):
            if isinstance(node, tuple):
                flush()
                operation = fragment(self.builder, node, kind="joined native OpenMP", native_metadata=self.native_metadata,
                                     native_host_reads=self.native_host_reads)
                self.native.append(operation)
                self.operation_count += len(operation.summary["operations"])
                result.append(operation)
                continue
            kind = _kind(node)
            if kind == "Comment" and not str(node).lstrip().lower().startswith("!$omp"):
                continue
            if kind == "Call_Stmt":
                call = self.builder.resolve(self.builder.entry, node)
                coordinator = self.builder.coordinator(call.procedure) if call.region is None else None
                if coordinator is not None:
                    flush()
                    self.calls.append(call)
                    result.append(Delegated(call, coordinator))
                    continue
                if call.region is not None and (call.region.runtime_guards or call.region.requires_numerical_environment):
                    flush()
                    private = {self.builder.entry.scope.bindings[name].root
                               for name in (*call.region.private_arrays, *call.region.private_scalars)}
                    try:
                        fallback = fragment(self.builder, self.builder.inline.original_selection(node),
                                            kind="guarded original numerical fallback", private_roots=private)
                    except CompilationError as error:
                        raise CompilationError("reached numerical guard lacks safe original native effects: " + str(error)) from error
                    self.guarded[id(call.node)] = fallback
                    self.native.append(fallback)
                self.calls.append(call)
                run.append(call)
                if id(call.node) in self.guarded:
                    flush()
                continue
            flush()
            if kind == "If_Construct":
                alternatives, body, header = [], [], None
                for item in node.content:
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if header is not None:
                            condition = None if _kind(header) == "Else_Stmt" else condition_fragment(
                                self.builder, header, native_host_reads=self.native_host_reads)
                            if condition is not None:
                                self.native.append(condition)
                                self.operation_count += len(condition.summary["operations"])
                            alternatives.append((condition, self.parse(body, depth + 1)))
                        header, body = item, []
                    else:
                        body.append(item)
                result.append(Branch(alternatives, statement_span(node)))
            elif kind == "If_Stmt":
                condition = condition_fragment(self.builder, node, native_host_reads=self.native_host_reads)
                self.native.append(condition)
                self.operation_count += len(condition.summary["operations"])
                action = self.builder.inline.statement_projection(node.items[1], node)
                result.append(Branch([(condition, self.parse((action,), depth + 1))], statement_span(node)))
            elif kind == "Associate_Construct":
                originals = self.builder.inline.original_selection(node)
                if len(originals) != 1 or _kind(originals[0]) != kind:
                    raise CompilationError("lexical ASSOCIATE requires one original source construct")
                record = self.builder.analysis._associate_scopes.get(id(originals[0]))
                if record is None or not record.available:
                    raise CompilationError("structured ASSOCIATE boundary: " +
                                           (record.reason if record is not None else "missing original selector proof"))
                result.append(Association(self.parse(node.content[1:-1], depth + 1),
                                          statement_span(node), record.public()))
            elif kind in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct"}:
                operation = fragment(self.builder, (node,), native_metadata=self.native_metadata,
                                     native_host_reads=self.native_host_reads)
                self.native.append(operation)
                self.operation_count += len(operation.summary["operations"])
                result.append(operation)
            else:
                raise CompilationError("structured ownership boundary: " + kind)
        flush()
        return result

    def inputs(self, arrays, scalars, written):
        for operation in self.native:
            for root, binding in operation.bindings.items():
                if root in operation.host_only_reads:
                    continue
                if binding.rank:
                    self.builder.capture(binding)
                    arrays[root] = binding
                else:
                    if binding.attributes & {"allocatable", "pointer", "optional", "volatile", "asynchronous", "value"}:
                        raise CompilationError("uncertain structured scalar capture: " + root)
                    scalars[root] = binding

            written.update(root for root, actions in operation.effects.items() if "write" in actions)
        # Hidden query controls are original module/local scalars too. Capturing
        # them preserves mutability rather than hoisting their current values.
        for call in self.calls:
            leaves, _ = self.builder.closure(call.procedure)
            if leaves:
                mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                try:
                    _, controls = self.builder.planning_inputs(call.procedure, mapping, require_estimates=False)
                except CompilationError:
                    continue
                for root in controls:
                    binding = self.builder.visible_binding(self.builder.entry, root)
                    scalars[root] = binding

    def check_conservative_definitions(self, arrays):
        # Unknown footprints retain full-array requirements. A partial source
        # definition cannot satisfy those merely because an unrelated point is
        # valid. End ownership before such native work rather than discovering
        # an unavoidable coherence failure after an earlier GPU segment.
        cleared = {root for call in self.calls for root in self.builder.roots_for(call)[1]}

        def require(actions, overwrites, exact):
            if exact:
                return
            for root in actions:
                if root not in overwrites and (self.builder.capture(arrays[root])["initialized"] != "whole"
                                               or root in cleared):
                    raise CompilationError("unknown native footprint requires complete definitions; ownership boundary: " + root)
        for operation in self.native:
            require(operation.coherence_effects, operation.overwrites, operation.coherence_sections.available)
        for call in self.calls:
            if not self.builder.closure(call.procedure)[0]:
                actions, _, overwrites = self.builder.roots_for(call)
                require(actions, overwrites, self.builder.analysis.native_sections(call.procedure).available)

    def prepare(self, arrays):
        """Split only where a later query needs values changed by earlier work."""
        def visit(items):
            result = []
            for item in items:
                if isinstance(item, Branch):
                    item.alternatives = [(condition, visit(body)) for condition, body in item.alternatives]
                    result.append(item)
                elif isinstance(item, Association):
                    item.body = visit(item.body)
                    result.append(item)
                elif isinstance(item, Segment):
                    run = []
                    for call in item.calls:
                        if id(call.node) in self.guarded:
                            if run:
                                result.append(self.segment(run, arrays))
                                run = []
                            result.append(self.segment((call,), arrays))
                            continue
                        candidate = [*run, call]
                        written = {root for c in candidate for root, actions in self.builder.roots_for(c)[0].items()
                                   if "write" in actions}
                        query = self.builder.owner_query(candidate, arrays, written)
                        if run and not query[0]:
                            result.append(self.segment(run, arrays))
                            run = []
                        run.append(call)
                    if run:
                        result.append(self.segment(run, arrays))
                else:
                    result.append(item)
            return result
        self.tree = visit(self.tree)
        cleared = {root for call in self.calls for root in self.builder.roots_for(call)[1]}
        if (self.builder.config.scope_execution != "reached"
                and any(root in cleared for segment in self.segments for root in segment.payload)):
            raise CompilationError("planning payload publication requires complete definitions across ownership")

    def segment(self, calls, arrays):
        written = {root for call in calls for root, actions in self.builder.roots_for(call)[0].items()
                   if "write" in actions}
        payload = set()
        for call in calls:
            if self.builder.closure(call.procedure)[0]:
                mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                try:
                    own, _ = self.builder.planning_inputs(call.procedure, mapping, require_estimates=False)
                    payload.update(own)
                except CompilationError:
                    pass
        item = Segment(list(calls), len(self.segments), self.builder.owner_query(calls, arrays, written), tuple(sorted(payload)))
        self.segments.append(item)
        return item

    def original(self, parameters):
        return [line for original in self.builder.inline.original_selection(self.nodes)
                for line in renamed(self.builder, original, parameters).splitlines()]

    def public(self):
        from compiler.scopes.source import _span

        def reduction_diagnostic(operation):
            if self.builder.config.scope_execution != "reached":
                return self.builder.analysis.reduction_candidates(
                    self.builder.entry.qualified, operation.summary.get("selected_node_ids", ()))
            # Serialization must not expand another proof search. Explicit
            # reduction analysis remains a separate source API until execution
            # requests such a proof at the reached operation.
            return {"analysis": "not_requested", "execution_supported": False,
                    "reason": "diagnostic serialization does not request reduction eligibility"}

        def tree(items):
            result = []
            for item in items:
                if isinstance(item, Segment):
                    result.append({"kind": "planning_segment", "segment_id": item.identity})
                elif isinstance(item, Branch):
                    result.append({"kind": "branch", "first_line": item.span[0], "last_line": item.span[1],
                                   "alternatives": [{"condition_operation": self.native.index(condition) if condition else None,
                                                     "guard": str(condition.nodes[0]) if condition else None,
                                                     "nodes": tree(body)} for condition, body in item.alternatives]})
                elif isinstance(item, Association):
                    result.append({"kind": "lexical_association", "first_line": item.span[0],
                                   "last_line": item.span[1], "proof": item.proof,
                                   "nodes": tree(item.body)})
                elif isinstance(item, Delegated):
                    available, reason = item.coordinator.scope.planning_status()
                    result.append({"kind": "reached_source_coordinator", "procedure": item.call.procedure,
                                   "first_line": item.call.node.item.span[0], "last_line": item.call.node.item.span[1],
                                   "structured_summary_identity": item.coordinator.public()["structured_summary_identity"],
                                   "estimate_available": available, "planning_reason": reason,
                                   "requirements": item.coordinator.public()["requirements"]})
                else:
                    result.append({"kind": "native_operation", "operation_id": self.native.index(item)})
            return result

        return {
            "ownership": {"lifetime": "one invocation", "planning_mode": "continuation", "close_count": 1,
                          "end_reason": "end of bounded source span; publish host-visible outputs and close"},
            "planning_segments": [
                {"segment_id": segment.identity, "first_line": _span(segment.calls[0].node)[0],
                 "last_line": _span(segment.calls[-1].node)[1], "calls": [call.procedure for call in segment.calls],
                 "gpu_leaves": sorted({leaf for call in segment.calls for leaf in self.builder.closure(call.procedure)[0]}),
                 "query_available": segment.query[0], "query_reason": segment.query[1],
                 "estimate_available": segment.query[2], "planning_reason": segment.query[3],
                 "query_host_reads": list(segment.payload),
                 "semantic_guards": [guard for call in segment.calls if call.region is not None
                                     for guard in (*call.region.runtime_guards,
                                                   *( ("fort_scope_numerical_environment_supported() /= 0",)
                                                      if call.region.requires_numerical_environment else ()))],
                 "guard_failure": "original native operation with coherence hooks; earlier work is retained",
                 "position": "when reached after preceding source operations"}
                for segment in self.segments],
            "native_operations": [{"operation_id": index, "kind": operation.kind,
                                   "first_line": operation.span[0], "last_line": operation.span[1],
                                   "resources": sorted(operation.effects),
                                   "resource_effects": {root: sorted(actions) for root, actions in sorted(operation.effects.items())},
                                   "managed_resources": sorted(operation.coherence_effects),
                                   "host_only_native_reads": operation.host_only_proof or None,
                                   "private_resources": sorted(operation.private_bindings),
                                   "completion": operation.summary["native_completion"],
                                   "sections": operation.sections.public(),
                                   "structured_summary_identity": operation.summary.get("structured_identity"),
                                   "demand_identity": operation.summary.get("demand_identity"),
                                   "selected_original_nodes": operation.summary.get("selected_node_ids", []),
                                   "atomic_effects": operation.summary.get("native_atomic"),
                                   "preserves_original_source": operation.preserve_original,
                                   "host_metadata": {root: {'descriptor': binding.public(),
                                       'coherence': 'opaque host-only range; registration rejects managed aliases',
                                       'device_capture': False}
                                       for root, binding in operation.host_metadata.items()},
                                   "reduction_analysis": reduction_diagnostic(operation),
                                   "planned_effects": True} for index, operation in enumerate(self.native)],
            "structured_tree": {"branch_depth_limit": 8, "operation_count": self.operation_count,
                                "nodes": tree(self.tree)},
        }

    def planning_status(self):
        statuses = [(segment.query[2], segment.query[3]) for segment in self.segments
                    if any(self.builder.closure(call.procedure)[0] for call in segment.calls)]

        def children(items):
            for item in items:
                if isinstance(item, Delegated):
                    statuses.append(item.coordinator.scope.planning_status())
                elif isinstance(item, Branch):
                    for _condition, body in item.alternatives:
                        children(body)
                elif isinstance(item, Association):
                    children(item.body)
        children(self.tree)
        return all(available for available, _reason in statuses), next(
            (reason for available, reason in statuses if not available), None)

    def emit(self, handles, parameters, actuals, imports, *, selector, execution_mode=None, terminal_owner=True,
             logical_lower_bounds=None, selector_name="fort_choose", preflight_failure=None, surround_native=False,
             provenance_segment=None):
        from compiler.scopes.provenance import position, source_id
        from compiler.scopes.source import _call, _checked, _name, _span
        builder = self.builder
        trace_segment = provenance_segment or source_id(builder, "segment", span=(self.first, self.last))

        def refined(operation, *, query):
            from compiler.scopes.access import build_native_accesses
            try:
                if operation.coherence_sections is None:
                    return None
                return build_native_accesses(operation.coherence_sections, handles,
                                             _name("fort_piece_", str(self.native.index(operation))),
                                             parameters={root: parameters[root] if root in parameters else builder.visible(builder.entry, root)
                                                         for root, binding in operation.bindings.items()
                                                         if not binding.rank and binding.dtype == "integer"},
                                             logical_lower_bounds=logical_lower_bounds,
                                             on_error=("exit " + _name("fort_prepare_", str(self.native.index(operation))),) if query else
                                             ("error stop 'structured native section preparation failed'",))
            except CompilationError:
                return None

        def choose():
            lines = ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)"]
            if builder.config.policy == "auto":
                lines += [f"if (fort_status == FORT_SCOPE_OK) fort_status = {selector_name}(fort_context, fort_decision)"]
            return lines

        def native_hooks(operation):
            identity = source_id(builder, "native_operation", span=operation.span or (self.first, self.last),
                                 operation_kind=operation.kind)
            implementation = source_id(builder, "implementation", span=operation.span or (self.first, self.last),
                                       backend="original_native")
            lines = position(builder, segment=trace_segment, operation=identity, implementation=implementation)
            accesses = refined(operation, query=False)
            if accesses is not None:
                lines += ["block", *[line for access in accesses for line in access.specification]]
                for access in accesses:
                    lines += [*access.prepare,
                              *_checked(f"fort_scope_host_begin(fort_context, {access.handle}, {access.access_name})")]
                return lines
            for root, actions in sorted(operation.coherence_effects.items()):
                flags = (["FORT_SCOPE_READ_ALL"] if "read" in actions else [])
                flags += ["FORT_SCOPE_WRITE_ALL"] if "write" in actions else []
                flags += ["FORT_SCOPE_OVERWRITE_ALL"] if root in operation.overwrites else []
                lines += ["fort_access = fort_scope_access()", "fort_access%flags = " + " + ".join(flags)]
                lines += _checked(f"fort_scope_host_begin(fort_context, {handles[root]}, fort_access)")
            return lines

        def native_end(operation):
            accesses = refined(operation, query=False)
            roots = ([access.handle for access in accesses] if accesses is not None
                     else [handles[root] for root in sorted(operation.coherence_effects)])
            return [*[line for handle in roots for line in _checked(f"fort_scope_host_end(fort_context, {handle})")],
                    *(["end block"] if accesses is not None else [])]

        def native_plan(operation):
            lines = ["fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"]
            accesses = refined(operation, query=True)
            if accesses is not None:
                block = _name("fort_prepare_", str(self.native.index(operation)))
                lines += [block + ": block", *[line for access in accesses for line in access.specification]]
                for index, access in enumerate(accesses, 1):
                    lines += [*access.prepare, f"fort_bindings({index}) = fort_scope_plan_binding()",
                              f"fort_bindings({index})%buffer = {access.handle}",
                              f"fort_bindings({index})%access = {access.access_name}"]
            for index, (root, actions) in enumerate(sorted(operation.coherence_effects.items()) if accesses is None else (), 1):
                flags = (["FORT_SCOPE_READ_ALL"] if "read" in actions else [])
                flags += ["FORT_SCOPE_WRITE_ALL"] if "write" in actions else []
                flags += ["FORT_SCOPE_OVERWRITE_ALL"] if root in operation.overwrites else []
                lines += [f"fort_bindings({index}) = fort_scope_plan_binding()",
                          f"fort_bindings({index})%buffer = {handles[root]}",
                          f"fort_bindings({index})%access%flags = " + " + ".join(flags)]
            count = len(accesses) if accesses is not None else len(operation.coherence_effects)
            lines += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_add( &",
                      "    fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &",
                      f"    {'c_loc(fort_bindings)' if count else 'c_null_ptr'}, {count}_c_size_t, &",
                      "    0.0_c_double, 0.0_c_double, 0_c_int)",
                      *(["end block " + block] if accesses is not None else []), *choose(),
                      "if (fort_status /= FORT_SCOPE_OK) then",
                      *(preflight_failure if preflight_failure is not None else
                        _checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)")), "endif"]
            return lines

        def execute(calls, mode, *, terminal=False):
            return builder.owner_calls(calls, mode, handles, parameters, actuals, imports, terminal=terminal)

        def record_segment(item, *, terminal=False):
            if not item.query[0]:
                if preflight_failure is not None:
                    return ["! Reached query unavailable: " + item.query[1], *preflight_failure]
                return ["! Current segment has no safe query: " + item.query[1],
                        *_checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"),
                        *execute(item.calls, "0_c_int", terminal=terminal), *_checked("fort_scope_wait(fort_context)")]
            lines = []
            if item.payload:
                publication = Native((), {root: {"read"} for root in item.payload}, set(), {}, {}, "query payload read")
                lines += [*native_plan(publication), *native_hooks(publication), *native_end(publication),
                          *_checked("fort_scope_wait(fort_context)")]
            lines += ["fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"]
            for call in item.calls:
                mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                native_handles = {formal: handles[root] for formal, root in mapping.items() if root in handles}
                native_handles.update({root: handle for root, handle in handles.items()
                                       if not root.startswith("argument::")})
                leaves, _ = builder.closure(call.procedure)
                lines += ["if (fort_status == FORT_SCOPE_OK) then"]
                if call.region is not None:
                    lines += builder.inline.emit_call(call, handles, parameters, imports, query=True)
                elif leaves:
                    query, roots = builder.query_clone(call.procedure)
                    module = builder.analysis.routines[call.procedure].scope.module
                    if module != builder.entry.scope.module:
                        imports.append(f"use {module}, only: {query}")
                    lines += _call(query, ["fort_context", *actuals(call),
                                           *[native_handles[root] for root in roots], "fort_status"])
                else:
                    block = _name("fort_record_", str(_span(call.node)))
                    query_lines = builder.native_plan(call, *builder.native_effects(call.procedure), native_handles,
                                                      actuals=actuals(call))
                    lines += [block + ": block", *[line.replace("return", "exit " + block)
                                                   if "return" in line else line for line in query_lines],
                              "end block " + block]
                lines += ["endif"]
            lines += choose()
            lines += ["if (fort_status /= FORT_SCOPE_OK) then",
                      *(preflight_failure if preflight_failure is not None else [
                        *_checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"),
                        *execute(item.calls, "0_c_int", terminal=terminal)]),
                      "else", *execute(item.calls, "fort_mode", terminal=terminal), "endif",
                      *_checked("fort_scope_wait(fort_context)")]
            return lines

        def segment_impl(item, *, terminal=False):
            tag = position(builder, segment=trace_segment, operation=source_id(builder, "numerical_segment",
                span=(_span(item.calls[0].node)[0], _span(item.calls[-1].node)[1]), segment_id=item.identity))
            if execution_mode is None:
                return [*tag, *record_segment(item, terminal=terminal)]
            return [*tag, "if (" + execution_mode + " == 0_c_int) then",
                    *_checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"),
                    *execute(item.calls, "0_c_int", terminal=terminal), *_checked("fort_scope_wait(fort_context)"),
                    "else", *record_segment(item, terminal=terminal), "endif"]

        def segment(item, *, terminal=False):
            guarded = [call for call in item.calls if id(call.node) in self.guarded]
            if not guarded:
                return segment_impl(item, terminal=terminal)
            if len(item.calls) != 1:
                raise CompilationError("guarded numerical work must form its own reached planning segment")
            call = guarded[0]
            operation = self.guarded[id(call.node)]
            guards = (*call.region.runtime_guards,
                      *(("fort_scope_numerical_environment_supported() /= 0",) if call.region.requires_numerical_environment else ()))
            lines = ["fort_numerical_guard = .true."]
            original = builder.inline.original_selection(call.node)
            scope = builder.analysis.source_scope_for(original[0], builder.entry.scope)
            for guard in guards:
                expression = F.Assignment_Stmt("fort_numerical_guard = " + guard).items[2]
                lines += ["if (fort_numerical_guard) fort_numerical_guard = "
                          + renamed(builder, expression, parameters, lexical_scope=scope)]
            lines += ["if (fort_numerical_guard) then", *segment_impl(item, terminal=terminal), "else",
                      *native_plan(operation), *native_hooks(operation)]
            lines += [line for original in operation.nodes for line in renamed(builder, original, parameters).splitlines()]
            return [*lines, *native_end(operation), *_checked("fort_scope_wait(fort_context)"), "endif"]

        def branch(item, index=0, *, terminal=False):
            condition, body = item.alternatives[index]
            if condition is None:
                return emit(body, terminal=terminal)
            lines = [*native_plan(condition), *native_hooks(condition),
                     "fort_branch = " + renamed(builder, condition.nodes[0], parameters), *native_end(condition),
                     *_checked("fort_scope_wait(fort_context)"),
                     "if (fort_branch) then", *emit(body, terminal=terminal)]
            if index + 1 < len(item.alternatives):
                lines += ["else", *branch(item, index + 1, terminal=terminal)]
            return [*lines, "endif"]

        def emit(items, *, terminal=False):
            lines = []
            for index, item in enumerate(items):
                closing = terminal and index == len(items)-1
                if isinstance(item, Segment):
                    lines += segment(item, terminal=closing)
                elif isinstance(item, Branch):
                    lines += branch(item, terminal=closing)
                elif isinstance(item, Association):
                    # Proved scalar variable selectors establish aliases only;
                    # each body reference already names its canonical storage.
                    lines += ["block", *emit(item.body, terminal=closing), "end block"]
                elif isinstance(item, Delegated):
                    lines += builder.owner_calls((item.call,), execution_mode or "fort_mode", handles, parameters,
                                                 actuals, imports, batch=False, terminal=False)
                    lines += _checked("fort_scope_wait(fort_context)")
                else:
                    lines += [*native_plan(item), *native_hooks(item)]
                    lines += [line for node in item.nodes for line in renamed(builder, node, parameters,
                              native_metadata=bool(item.host_metadata)).splitlines()]
                    lines += [*native_end(item), *_checked("fort_scope_wait(fort_context)")]
            return lines

        if surround_native:
            if (len(self.tree) != 1 or not isinstance(self.tree[0], Native)
                    or not self.tree[0].preserve_original or refined(self.tree[0], query=False) is not None):
                raise CompilationError("original native hooks require one complete unrefined source operation")
            operation = self.tree[0]
            return ([*native_plan(operation), *native_hooks(operation)],
                    [*native_end(operation), *_checked("fort_scope_wait(fort_context)")])
        return emit(self.tree, terminal=terminal_owner)
