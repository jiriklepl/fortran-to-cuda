"""Bounded structured owners, with queries evaluated at reached source points.

This is source scheduling, not numerical lowering. Inline source operations stay
native and retain their original OpenMP team. Numerical calls reuse the existing
mode-bearing workers and public planning interface.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.source_effects import Binding, _kind
from compiler.ir import CompilationError


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


def joined_group_completion(builder, nodes):
    """Prove completion of one bounded original PARALLEL region.

    Its end barrier joins even NOWAIT worksharing loops. Only numeric local
    scalar PRIVATE clauses and explicit SHARED names are admitted; tasks,
    detached work and nested teams remain boundaries.
    """
    if not nodes or directive(nodes[-1]) != "end parallel":
        raise CompilationError("native parallel operation requires a joined END PARALLEL")

    def clauses(text):
        remainder = text.strip()
        while remainder:
            match = re.match(r"(private|shared|default|collapse|schedule)\s*\(([^()]*)\)\s*", remainder)
            if match is None:
                raise CompilationError("unsupported joined native OpenMP clause: " + remainder)
            kind, values = match.group(1), match.group(2).strip()
            if kind == "default":
                if values not in {"shared", "none"}:
                    raise CompilationError("unsupported native OpenMP DEFAULT")
            elif kind == "collapse":
                if values not in {"1", "2", "3", "4"}:
                    raise CompilationError("unsupported native OpenMP COLLAPSE")
            elif kind == "schedule":
                if values != "static":
                    raise CompilationError("unsupported native OpenMP SCHEDULE")
            else:
                for name in values.split(","):
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name.strip()):
                        raise CompilationError("native OpenMP clause needs resolved variable names")
                    binding = builder.analysis._binding(builder.entry.scope, name.strip())
                    if binding is None:
                        raise CompilationError("unresolved native OpenMP clause variable")
                    if kind == "private" and (binding.rank or "save" in binding.attributes
                                              or not binding.root.startswith(builder.entry.qualified + "::")):
                        raise CompilationError("native PRIVATE requires an original local scalar")
            remainder = remainder[match.end():].lstrip(", ")

    first = directive(nodes[0])
    if first is None or not re.match(r"parallel(?:\s|$)", first):
        raise CompilationError("native operation is not one complete PARALLEL region")
    clauses(first[len("parallel"):])
    index = 1
    while index < len(nodes) - 1:
        node, text = nodes[index], directive(nodes[index])
        if text is None:
            if _kind(node) != "Comment":
                raise CompilationError("native PARALLEL body requires bounded worksharing DO loops")
            index += 1
            continue
        if not re.match(r"do(?:\s|$)", text):
            raise CompilationError("unsupported joined native OpenMP directive: " + text)
        clauses(text[2:])
        index += 1
        while index < len(nodes) and _kind(nodes[index]) == "Comment" and directive(nodes[index]) is None:
            index += 1
        if index >= len(nodes) or _kind(nodes[index]) != "Block_Nonlabel_Do_Construct":
            raise CompilationError("native OpenMP DO requires one complete associated loop")
        if any(directive(item) is not None for item in walk(nodes[index])):
            raise CompilationError("nested OpenMP directives inside a native loop are unsupported")
        index += 1
        while index < len(nodes) and _kind(nodes[index]) == "Comment" and directive(nodes[index]) is None:
            index += 1
        if index >= len(nodes) or directive(nodes[index]) not in {"end do", "end do nowait"}:
            raise CompilationError("native OpenMP DO requires a matching END DO")
        index += 1
    return {"available": True, "reason": "original bounded PARALLEL region joins at END PARALLEL",
            "caller_contract": "serial_source_scope", "requires_serial_caller": True,
            "has_openmp_in_closure": True, "has_opaque_calls_in_closure": False}


def grouped_nodes(nodes):
    """Keep whole joined groups as one native scheduling operation."""
    expanded = []
    for node in nodes:
        if _kind(node) == "Block_Nonlabel_Do_Construct":
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
            end = index + 1
            while end < len(nodes) and directive(nodes[end]) != "end parallel":
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


def fragment(builder, nodes, *, kind="native source"):
    """Analyze original typed nodes with original bindings and a private cache.

    The fragment is already inside the entry: its old dummy INTENT(OUT) and
    specification expressions must not execute a second time. The original
    routine/analysis remain untouched, including source and contract identities.
    """
    analysis = copy.copy(builder.analysis)
    analysis.routines = dict(analysis.routines)
    analysis.summaries, analysis._closures, analysis._native_sections = {}, {}, {}
    routine = copy.copy(builder.entry)
    routine.scope = copy.copy(routine.scope)
    routine.scope.bindings = {name: copy.copy(binding) for name, binding in routine.scope.bindings.items()}
    for binding in routine.scope.bindings.values():
        binding.intent = None
        if binding.rank and "allocatable" in binding.attributes and binding.root.startswith("argument::"):
            # Original caller guards + source-bound lifetime facts authorize
            # borrowing this descriptor within the fragment, not allocation
            # changes or allocatable child interfaces. Whole allocatable writes
            # remain boundaries in the existing effect analysis.
            builder.capture(binding)
            analysis._stable_module_allocatables = analysis._stable_module_allocatables | {binding.root}
    if kind == "condition read":
        probe = "fort_condition_projection"
        routine.scope.bindings[probe] = Binding(probe, routine.qualified + "::" + probe, "logical", 4, 0)
    routine.scope.node = copy.copy(routine.scope.node)
    routine.execution = copy.copy(routine.execution)
    routine.execution.content = list(nodes)
    routine.scope.node.content = [routine.execution]
    # Entry-local resources can also have registered device state from an
    # earlier call. Treat their fragment accesses as externally visible while
    # removing procedure-entry definition events from this inner operation.
    routine.arguments, routine.issues = tuple(routine.scope.bindings), []
    analysis.routines[routine.qualified] = routine
    summary = analysis.summarize(routine.qualified)
    if not summary["complete"]:
        raise CompilationError("structured native effects incomplete: " + "; ".join(summary["reasons"]))
    if any(operation["kind"] == "boundary" or (operation["kind"] == "control"
           and not (kind == "condition read" and operation["source"].upper() == "CONTINUE"))
           for operation in summary["operations"]):
        raise CompilationError("unsupported exit or lifetime boundary inside structured ownership")
    completion = summary["native_completion"]
    if kind == "joined native OpenMP":
        completion = summary["native_completion"] = joined_group_completion(builder, nodes)
    if not completion["available"]:
        raise CompilationError("structured native completion unproven: " + completion["reason"])
    effects = {}
    for operation in summary["operations"]:
        if operation["kind"] in {"call", "native_contract"}:
            raise CompilationError("inline native calls require a separately mapped source operation")
        if operation["kind"] in {"read", "write", "overwrite"} and operation["rank"]:
            effects.setdefault(operation["resource"], set()).add(
                "read" if operation["kind"] == "read" else "write")
    bindings = {}
    for node in nodes:
        for name in walk(node, F.Name):
            binding = builder.analysis._binding(builder.entry.scope, name)
            if binding is not None:
                bindings[binding.root] = binding
    span = (statement_span(nodes[0])[0], statement_span(nodes[-1])[1]) if kind != "condition read" else ()
    return Native(tuple(nodes), effects, set(summary["guaranteed_whole_overwrites"]), bindings, summary, kind,
                  analysis.native_sections(routine.qualified), span)


def condition_fragment(builder, header):
    # A typed local assignment projection also allows the ordinary native
    # section analyzer to prove exact point/rectangular condition reads.
    condition = header.items[0]
    projection = F.Assignment_Stmt("fort_condition_projection = " + str(condition))
    result = fragment(builder, (projection,), kind="condition read")
    result.nodes = (condition,)
    result.span = statement_span(header)
    return result


def renamed(builder, node, parameters):
    """Rewrite resolved variable AST nodes, never string substrings."""
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
        if (_kind(value) == "Intrinsic_Function_Reference" and str(value.items[0]).lower() == "allocated"):
            arguments = getattr(value.items[1], "items", ())
            if len(arguments) == 1 and _kind(arguments[0]) == "Name":
                binding = builder.analysis._binding(builder.entry.scope, arguments[0])
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
    clone = clone_ast(node)
    keywords = {id(item.items[0]) for item in walk(clone, F.Actual_Arg_Spec)}
    for name in walk(clone, F.Name):
        if id(name) in keywords:
            continue  # Keywords name callee formals, not caller storage.
        binding = builder.analysis._binding(builder.entry.scope, name)
        if binding is not None and binding.root in parameters:
            name.string = parameters[binding.root]
    return str(clone)


class StructuredScope:
    def __init__(self, builder, nodes):
        self.builder, self.nodes = builder, tuple(nodes)
        self.calls, self.native, self.segments = [], [], []
        self.operation_count = 0
        self.tree = self.parse(nodes)
        if len(self.calls) > 32 or self.operation_count > 256:
            raise CompilationError("bounded structured owner exceeds call/operation budget")
        self.first, self.last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
        original = builder.entry.scope.path.read_text().splitlines()[self.first - 1:self.last]
        if any(line.lstrip().startswith("#") for line in original):
            raise CompilationError("preprocessor control boundary inside structured ownership")

    def parse(self, nodes, depth=0):
        if depth > 8:
            raise CompilationError("bounded structured owner exceeds branch depth")
        result, run = [], []

        def flush():
            if run:
                result.append(Segment(list(run)))
                run.clear()

        for node in grouped_nodes(nodes):
            if isinstance(node, tuple):
                flush()
                operation = fragment(self.builder, node, kind="joined native OpenMP")
                self.native.append(operation)
                self.operation_count += len(operation.summary["operations"])
                result.append(operation)
                continue
            kind = _kind(node)
            if kind == "Comment" and not str(node).lstrip().lower().startswith("!$omp"):
                continue
            if kind == "Call_Stmt":
                call = self.builder.resolve(self.builder.entry, node)
                self.calls.append(call)
                run.append(call)
                continue
            flush()
            if kind == "If_Construct":
                alternatives, body, header = [], [], None
                for item in node.content:
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if header is not None:
                            condition = None if _kind(header) == "Else_Stmt" else condition_fragment(self.builder, header)
                            if condition is not None:
                                self.native.append(condition)
                                self.operation_count += len(condition.summary["operations"])
                            alternatives.append((condition, self.parse(body, depth + 1)))
                        header, body = item, []
                    else:
                        body.append(item)
                result.append(Branch(alternatives, statement_span(node)))
            elif kind == "If_Stmt":
                condition = condition_fragment(self.builder, node)
                self.native.append(condition)
                self.operation_count += len(condition.summary["operations"])
                action = copy.copy(node.items[1])
                if getattr(action, "item", None) is None:
                    action.item = node.item
                result.append(Branch([(condition, self.parse((action,), depth + 1))], statement_span(node)))
            elif kind in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct"}:
                operation = fragment(self.builder, (node,))
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
                    binding = self.builder.analysis._binding(self.builder.entry.scope,
                                                             self.builder.visible(self.builder.entry, root))
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
            require(operation.effects, operation.overwrites, operation.sections.available)
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
                elif isinstance(item, Segment):
                    run = []
                    for call in item.calls:
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
        if any(root in cleared for segment in self.segments for root in segment.payload):
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
        return [line for node in self.nodes for line in renamed(self.builder, node, parameters).splitlines()]

    def public(self):
        from compiler.scopes.source import _span

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
                 "position": "when reached after preceding source operations"}
                for segment in self.segments],
            "native_operations": [{"operation_id": index, "kind": operation.kind,
                                   "first_line": operation.span[0], "last_line": operation.span[1],
                                   "resources": sorted(operation.effects),
                                   "completion": operation.summary["native_completion"],
                                   "sections": operation.sections.public(),
                                   "planned_effects": True} for index, operation in enumerate(self.native)],
            "structured_tree": {"branch_depth_limit": 8, "operation_count": self.operation_count,
                                "nodes": tree(self.tree)},
        }

    def emit(self, handles, parameters, actuals, imports, *, selector):
        from compiler.scopes.source import _call, _checked, _name, _span
        builder = self.builder

        def refined(operation, *, query):
            from compiler.scopes.access import build_native_accesses
            try:
                if operation.sections is None:
                    return None
                return build_native_accesses(operation.sections, handles,
                                             _name("fort_piece_", str(self.native.index(operation))),
                                             on_error=("exit " + _name("fort_prepare_", str(self.native.index(operation))),) if query else
                                             ("error stop 'structured native section preparation failed'",))
            except CompilationError:
                return None

        def choose():
            lines = ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)"]
            if builder.config.policy == "auto":
                lines += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_choose(fort_context, fort_decision)"]
            return lines

        def native_hooks(operation):
            lines = []
            accesses = refined(operation, query=False)
            if accesses is not None:
                lines += ["block", *[line for access in accesses for line in access.specification]]
                for access in accesses:
                    lines += [*access.prepare,
                              *_checked(f"fort_scope_host_begin(fort_context, {access.handle}, {access.access_name})")]
                return lines
            for root, actions in sorted(operation.effects.items()):
                flags = (["FORT_SCOPE_READ_ALL"] if "read" in actions else [])
                flags += ["FORT_SCOPE_WRITE_ALL"] if "write" in actions else []
                flags += ["FORT_SCOPE_OVERWRITE_ALL"] if root in operation.overwrites else []
                lines += ["fort_access = fort_scope_access()", "fort_access%flags = " + " + ".join(flags)]
                lines += _checked(f"fort_scope_host_begin(fort_context, {handles[root]}, fort_access)")
            return lines

        def native_end(operation):
            accesses = refined(operation, query=False)
            roots = [access.handle for access in accesses] if accesses is not None else [handles[root] for root in sorted(operation.effects)]
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
            for index, (root, actions) in enumerate(sorted(operation.effects.items()) if accesses is None else (), 1):
                flags = (["FORT_SCOPE_READ_ALL"] if "read" in actions else [])
                flags += ["FORT_SCOPE_WRITE_ALL"] if "write" in actions else []
                flags += ["FORT_SCOPE_OVERWRITE_ALL"] if root in operation.overwrites else []
                lines += [f"fort_bindings({index}) = fort_scope_plan_binding()",
                          f"fort_bindings({index})%buffer = {handles[root]}",
                          f"fort_bindings({index})%access%flags = " + " + ".join(flags)]
            count = len(accesses) if accesses is not None else len(operation.effects)
            lines += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_add( &",
                      "    fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &",
                      f"    {'c_loc(fort_bindings)' if count else 'c_null_ptr'}, {count}_c_size_t, &",
                      "    0.0_c_double, 0.0_c_double, 0_c_int)",
                      *(["end block " + block] if accesses is not None else []), *choose(),
                      "if (fort_status /= FORT_SCOPE_OK) then",
                      *_checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"), "endif"]
            return lines

        def execute(calls, mode, *, terminal=False):
            return builder.owner_calls(calls, mode, handles, parameters, actuals, imports, terminal=terminal)

        def segment(item, *, terminal=False):
            if not item.query[0]:
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
                if leaves:
                    query, roots = builder.query_clone(call.procedure)
                    module = builder.analysis.routines[call.procedure].scope.module
                    if module != builder.entry.scope.module:
                        imports.append(f"use {module}, only: {query}")
                    lines += _call(query, ["fort_context", *actuals(call),
                                           *[native_handles[root] for root in roots], "fort_status"])
                else:
                    block = _name("fort_record_", str(_span(call.node)))
                    query_lines = builder.native_plan(call, *builder.native_effects(call.procedure), native_handles)
                    lines += [block + ": block", *[line.replace("return", "exit " + block)
                                                   if "return" in line else line for line in query_lines],
                              "end block " + block]
                lines += ["endif"]
            lines += choose()
            lines += ["if (fort_status /= FORT_SCOPE_OK) then",
                      *_checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"),
                      *execute(item.calls, "0_c_int", terminal=terminal), "else", *execute(item.calls, "fort_mode", terminal=terminal), "endif",
                      *_checked("fort_scope_wait(fort_context)")]
            return lines

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
            for position, item in enumerate(items):
                closing = terminal and position == len(items)-1
                if isinstance(item, Segment):
                    lines += segment(item, terminal=closing)
                elif isinstance(item, Branch):
                    lines += branch(item, terminal=closing)
                else:
                    lines += [*native_plan(item), *native_hooks(item)]
                    lines += [line for node in item.nodes for line in renamed(builder, node, parameters).splitlines()]
                    lines += [*native_end(item), *_checked("fort_scope_wait(fort_context)")]
            return lines

        return emit(self.tree, terminal=True)
