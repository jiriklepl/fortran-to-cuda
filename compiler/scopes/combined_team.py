"""One original numerical team with an independently authorized native path.

Independently proved original native effects can replace an exact all-CPU
worker range without retiring future GPU choices. An installed query can hand
that range to the pristine original team before its fork; generated workers
retain their participant vote and preparation protocol.
"""

import re
from copy import copy
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.native_group_structure import NATIVE_GROUP_SCAN_LIMIT, original_sequence
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.joined_team import declared_names, ordinary_entry_guard, original_team_source
from compiler.scopes.segments import (
    Segment,
    StructuredScope,
    directive,
    fortran_lines,
    original_roots,
    statement_span,
)


def is_combined_group(builder, nodes):
    """Route original syntax only; the consumer separately proves authority."""
    originals = original_roots(builder.inline.original_selection(nodes))
    count, directives = 0, []
    for original in originals:
        for node in walk(original):
            count += 1
            if count > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("combined team routing scan budget exhausted")
            text = directive(node)
            if text is not None:
                directives.append((node, text))
    if not directives:
        return False
    if re.match(r"parallel\s+do(?:\s|$)", directives[0][1]):
        return True
    if not re.match(r"parallel(?:\s|$)", directives[0][1]):
        return False
    # Mixed/native groups retain their existing JoinedTeam route. This
    # recognizes source shape only, never numerical legality.
    try:
        items, units = _separate_units(originals)
    except CompilationError:
        return False
    expected = _separate_directives(items, units)
    return tuple(id(node) for node, _text in directives) == tuple(id(node) for node in expected)


def _separate_units(originals):
    """Recognize a flat whole PARALLEL without detaching a worksharing unit."""
    peeled = {}
    items = original_sequence(originals, peeled)
    if (not items or not re.match(r"parallel(?:\s|$)", directive(items[0]) or "")
            or directive(items[-1]) != "end parallel"):
        raise CompilationError("separate numerical team requires a complete original PARALLEL group")
    units, index = [], 1
    while index < len(items) - 1:
        opening = items[index]
        text = directive(opening)
        if _kind(opening) == "Comment" and text is None:
            index += 1
            continue
        if text is None or not re.match(r"do(?:\s|$)", text):
            raise CompilationError("separate numerical team requires only flat original worksharing DO units")
        index += 1
        while index < len(items) - 1 and _kind(items[index]) == "Comment" and directive(items[index]) is None:
            index += 1
        if index >= len(items) - 1 or _kind(items[index]) != "Block_Nonlabel_Do_Construct":
            raise CompilationError("separate numerical team requires complete original associated DO loops")
        loop = items[index]
        content = peeled.get(id(loop), tuple(loop.content))
        if (not content or _kind(content[0]) != "Nonlabel_Do_Stmt"
                or _kind(content[-1]) != "End_Do_Stmt"):
            raise CompilationError("separate numerical team requires complete original counted DO loops")
        index += 1
        while index < len(items) - 1 and _kind(items[index]) == "Comment" and directive(items[index]) is None:
            index += 1
        ending = None
        if index < len(items) - 1 and directive(items[index]) in {"end do", "end do nowait"}:
            ending = items[index]
            index += 1
        units.append((opening, loop, content[0], content[-1], ending))
        if len(units) > 32:
            raise CompilationError("separate numerical team source-unit budget exhausted")
    if not units:
        raise CompilationError("separate numerical team requires at least one original DO unit")
    return items, tuple(units)


def _separate_directives(items, units):
    return (items[0], *(node for unit in units for node in (unit[0], unit[-1]) if node is not None), items[-1])


def _clauses(builder, text, allowed):
    result, count = [], 0
    remainder = text.strip()
    while remainder:
        count += 1
        if count > builder.analysis.operation_limit:
            raise CompilationError("combined numerical team clause budget exhausted")
        match = re.match(r"(private|shared|default|collapse|schedule)\s*\(([^()]*)\)\s*", remainder)
        if match is None or match.group(1) not in allowed:
            raise CompilationError("combined numerical team has an unsupported clause partition: " + remainder)
        kind, value = match.group(1), match.group(2).strip()
        if ((kind == "default" and value not in {"none", "shared"})
                or (kind == "collapse" and value not in {"1", "2", "3", "4"})
                or (kind == "schedule" and value not in {"static", "runtime"})
                or (kind in {"private", "shared"} and any(
                    not re.fullmatch(r"[a-z][a-z0-9_]*", name.strip()) for name in value.split(",")))):
            raise CompilationError("combined numerical team clause operands are unsupported")
        result.append((kind, kind + "(" + value + ")"))
        remainder = remainder[match.end():].lstrip(", ")
    return tuple(result)


def _source_partition(builder, originals):
    """Split only the already checked finite combined-clause vocabulary."""
    directives, count = [], 0
    for original in originals:
        for node in walk(original):
            count += 1
            if count > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("combined team source scan budget exhausted")
            if (text := directive(node)) is not None:
                directives.append((node, text))
    if not directives or not re.match(r"parallel(?:\s|$)", directives[0][1]):
        raise CompilationError("combined numerical team requires one original PARALLEL group")
    opening, text = directives[0]
    separate = not re.match(r"parallel\s+do(?:\s|$)", text)
    if separate:
        items, units = _separate_units(originals)
        ending = items[-1]
        expected = _separate_directives(items, units)
        if tuple(id(node) for node, _text in directives) != tuple(id(node) for node in expected):
            raise CompilationError("separate numerical team cannot contain nested or unmatched directives")
        parallel = [clause for _kind_, clause in _clauses(builder, text[len("parallel"):],
                                                        {"private", "shared", "default"})]
        worksharing = []
        clauses = len(parallel)
        for unit in units:
            unit_clauses = _clauses(builder, directive(unit[0])[len("do"):], {"private", "collapse", "schedule"})
            clauses += len(unit_clauses)
        first, last = statement_span(opening)[0], statement_span(ending)[1]
        body_first, body_last = statement_span(opening)[1] + 1, statement_span(ending)[0] - 1
        boundaries = (opening, ending, *(node for unit in units for node in (unit[0], unit[2], unit[3], unit[4])
                                         if node is not None))
    else:
        first, last, body_first, body_last, boundaries, parallel, worksharing, clauses = _combined_partition(
            builder, originals, directives)
        units = (None,)
    if clauses > builder.analysis.operation_limit:
        raise CompilationError("combined numerical team clause budget exhausted")
    if not first < body_first <= body_last <= last:
        raise CompilationError("combined numerical team source association is unordered")
    source = builder.entry.scope.path
    captured = builder.analysis.source_text(source)
    if len(captured.encode()) > 32 * 1024**2:
        raise CompilationError("combined numerical team original source byte budget exhausted")
    lines = captured.splitlines(keepends=True)
    raw = "".join(lines[first-1:last])
    if (last > len(lines) or len(raw.encode()) > NATIVE_GROUP_SCAN_LIMIT
            or any(line.lstrip().startswith("#") for line in lines[first-1:last])):
        raise CompilationError("combined numerical team requires a bounded original mapped source span")
    if any(";" in line for node in boundaries for line in lines[statement_span(node)[0]-1:statement_span(node)[1]]):
        raise CompilationError("combined numerical team cannot replace a semicolon source boundary")
    return {"first": first, "last": last, "body": "".join(lines[body_first-1:body_last]),
            "source_sha256": sha256(raw.encode()).hexdigest(),
            "parallel_clauses": tuple(parallel), "do_clauses": tuple(worksharing),
            "group_kind": "separate_parallel" if separate else "combined_parallel_do", "unit_count": len(units),
            "scan_nodes": count, "clause_count": clauses}


def _combined_partition(builder, originals, directives):
    opening, text = directives[0]
    ending = None
    if len(directives) == 2 and directives[1][1] == "end parallel do":
        ending = directives[1][0]
    elif len(directives) != 1:
        raise CompilationError("combined numerical team cannot split nested or unmatched directives")
    loops = tuple(node for node in originals if _kind(node) == "Block_Nonlabel_Do_Construct")
    if len(loops) != 1:
        raise CompilationError("combined numerical team requires exactly one original associated loop")
    loop = loops[0]
    header = next((node for node in loop.content if _kind(node) == "Nonlabel_Do_Stmt"), None)
    end = loop.content[-1]
    if header is None or _kind(end) != "End_Do_Stmt":
        raise CompilationError("combined numerical team requires a complete original counted DO")
    first = statement_span(opening)[0]
    body_first, body_last = statement_span(header)[0], statement_span(end)[1]
    last = statement_span(ending)[1] if ending is not None else body_last
    boundaries = (opening, header, end, *((ending,) if ending is not None else ()))
    parallel, worksharing = [], []
    parsed = _clauses(builder, re.sub(r"^parallel\s+do\b", "", text, count=1),
                      {"private", "shared", "default", "collapse", "schedule"})
    for kind, clause in parsed:
        (worksharing if kind in {"collapse", "schedule"} else parallel).append(clause)
    return first, last, body_first, body_last, boundaries, parallel, worksharing, len(parsed)


def _check_generated_names(originals, specification, imports, generated, block):
    """A new enclosing BLOCK must not shadow an original imported name."""
    names = {name.lower() for name in declared_names(specification)}
    for line in specification:
        if "::" in line and re.search(r"\bparameter\b", line.split("::", 1)[0], re.I):
            # OpenMP sharing omits constants, but the enclosing BLOCK still
            # declares their names. Inspect declarators, never kind operands.
            names.update(name.lower() for name in declared_names(["integer :: " + line.split("::", 1)[1]]))
    names.update((block.lower(), "fort_context", "fort_returned", "fort_native_ready"))
    for line in imports:
        if "only:" in line.lower():
            for item in re.split(r"\bonly\s*:", line, maxsplit=1, flags=re.I)[1].split(","):
                names.add(item.split("=>", 1)[0].strip().lower())
    for line in (*specification, *generated):
        # These are emitted call/type/constant sites, excluding original DO
        # source. Compare full identifiers, never a source-name prefix rule.
        names.update(name.lower() for name in re.findall(r"\b(fort_[a-z0-9_]+)\s*\(", line, re.I))
        names.update(name.lower() for name in re.findall(r"\b(FORT_SCOPE_[A-Z0-9_]+)\b", line))
    count = 0
    for original in originals:
        for node in walk(original):
            count += 1
            if count > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("combined team generated-name scan budget exhausted")
            if _kind(node) == "Name" and str(node).lower() in names:
                raise CompilationError("combined numerical team generated name conflicts with original reference: "
                                       + str(node))
            if (text := directive(node)) is not None:
                for match in re.finditer(r"\b(?:private|shared)\s*\(([^()]*)\)", text):
                    for name in match.group(1).split(","):
                        if name.strip().lower() in names:
                            raise CompilationError(
                                "combined numerical team generated name conflicts with original reference: "
                                + name.strip())


def _narrow_runtime_imports(imports, specification, generated, *, original_imports=()):
    """Introduce only used exports, leaving unrelated original aliases intact."""
    path = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    if path.stat().st_size > NATIVE_GROUP_SCAN_LIMIT:
        raise CompilationError("combined team runtime namespace byte budget exhausted")
    source = path.read_text()
    if len(source.encode()) > NATIVE_GROUP_SCAN_LIMIT:
        raise CompilationError("combined team runtime namespace byte budget exhausted")
    if not re.search(r"^\s*module\s+fort_scoped_memory\s*$", source, re.I | re.M) or not re.search(
            r"^\s*private\s*$", source, re.I | re.M):
        raise CompilationError("combined team requires an explicit private runtime export namespace")
    exports, pending = set(), ""
    for line in source.splitlines():
        text = line.split("!", 1)[0].strip()
        if not text:
            continue
        text = pending + text.lstrip("& ")
        if text.endswith("&"):
            pending = text[:-1] + " "
            continue
        pending = ""
        prefix, separator, declarators = text.partition("::")
        if re.search(r"\bpublic\b", prefix, re.I):
            if not separator:
                raise CompilationError("combined team runtime export declaration is unsupported")
            exports.update(name.lower() for name in declared_names(["integer :: " + declarators]))
    if pending or not exports:
        raise CompilationError("combined team runtime export namespace is incomplete")
    # These standard C ABI entities cover existing scope preparation. Selecting
    # only referenced entities avoids importing unrelated intrinsic constants.
    iso = {"c_associated", "c_bool", "c_char", "c_double", "c_f_pointer", "c_float", "c_int",
           "c_int32_t", "c_int64_t", "c_intptr_t", "c_loc", "c_null_char", "c_null_ptr", "c_ptr", "c_size_t"}
    emitted = "\n".join((*specification, *generated)).lower()
    used = set(re.findall(r"\b[a-z][a-z0-9_]*\b", emitted))
    # A literal's kind follows an underscore, so it has no identifier word
    # boundary: 1_c_size_t and 0.0_c_double still need their ISO exports.
    used.update(re.findall(
        r"(?<![a-z0-9_])(?:\d+(?:\.\d*)?|\.\d+)(?:[ed][+-]?\d+)?_([a-z][a-z0-9_]*)\b", emitted))
    used.update(re.findall(r"\.(?:true|false)\._([a-z][a-z0-9_]*)\b", emitted))
    for index, statement in enumerate(imports):
        lower = statement.strip().lower()
        if lower in {"use fort_scoped_memory", "use iso_c_binding"} and statement not in original_imports:
            selected = sorted(used & (exports if lower.endswith("fort_scoped_memory") else iso))
            if not selected:
                raise CompilationError("combined team generated runtime import has no required exports")
            imports[index] = statement + ", only: " + ", ".join(selected)


class CombinedNumericalTeam(StructuredScope):
    """An exact numerical group with an optional independently proved CPU path."""

    def __init__(self, builder, nodes, *, owning_native=True):
        self.builder, self.nodes = builder, tuple(nodes)
        self.original_nodes = original_roots(builder.inline.original_selection(nodes))
        self.partition = _source_partition(builder, self.original_nodes)
        facade = builder.inline.prepare_combined_team(self.original_nodes)
        region = builder._consume_numerical_region(
            builder.analysis, builder.entry.qualified, self.original_nodes)
        if region is None:
            raise CompilationError("combined numerical team requires registered original preparation")
        self.prepared, self.effect = region.prepared, region.effects
        if not self.prepared.completion.get("has_openmp_in_closure"):
            raise CompilationError("combined numerical team requires original joined completion")
        self.first, self.last = self.partition["first"], self.partition["last"]
        call = builder.inline.call(facade)
        outlined = builder._consume_call_numerical_region(call)
        if outlined is None or outlined.prepared is not self.prepared or outlined.effects is not self.effect:
            raise CompilationError("combined numerical team requires exact registered physical effects")
        self.completion = self.prepared.completion_proof
        from compiler.scopes.source_reads import prepare_source_reads
        self.source_reads = prepare_source_reads(builder, call)
        self.native_metadata = True
        self.native_host_reads = False
        self.calls, self.native, self.segments = [call], [], []
        self.guarded = {}
        self.close_guards = {id(call.node): self.effect}
        self.tree = [Segment([call])]
        self.operation_count = self.effect.public()["ordered_effect_count"]
        if self.operation_count > builder.analysis.operation_limit:
            raise CompilationError("combined numerical team ordered effect budget exhausted")
        self.cpu_native, self.cpu_native_decline = None, None
        self.cpu_native_sections = None
        self.cpu_native_section_authority = None
        if owning_native and builder.config.scope_execution == "graph" and builder.config.policy == "auto":
            from compiler.scopes.access import build_native_accesses
            from compiler.scopes.segments import fragment

            try:
                operation = fragment(builder, self.original_nodes, kind="joined native OpenMP",
                                     retain_proof_analysis=True)
                if operation.host_metadata or operation.native_environment or operation.host_only_reads:
                    raise CompilationError("original CPU team requires no extra host metadata or native environment effects")
                if any(item.get("kind") in {"write", "overwrite"} and not item.get("rank", 0)
                       and item.get("resource") not in operation.private_bindings
                       for item in operation.summary["operations"]):
                    raise CompilationError("original CPU team has a shared scalar write")
                numerical_roots = {binding.root for binding in call.region.bindings if binding.rank}
                if not set(operation.coherence_effects) <= numerical_roots:
                    raise CompilationError("original CPU team requires storage outside its numerical interface")
                sections = operation.coherence_sections
                section_authority = "independent original native sections"
                if not sections.available:
                    # One original numerical worker has no internal worker
                    # ordering or overwrite union to reconstruct. Its issued
                    # physical effect step already describes the exact same
                    # complete source team under every typed precondition.
                    # Keep the independently derived source/completion/effect
                    # summary above; reuse only physical rectangle authority.
                    if (len(self.prepared.offload.units) != 1 or len(self.effect.steps) != 1
                            or not self.prepared.all_runtime_guards_typed):
                        raise CompilationError("original CPU team native sections: " + str(sections.reason))
                    sections = self.effect.steps[0].sections
                    by_root = {resource.resource: resource for resource in sections.resources}
                    for root, actions in operation.coherence_effects.items():
                        resource = by_root.get(root)
                        if resource is None or ("read" in actions and not resource.reads) or (
                                "write" in actions and not resource.writes):
                            raise CompilationError("single numerical worker omits original native storage effects: " + root)
                    if not set(by_root) <= numerical_roots:
                        raise CompilationError("single numerical worker native effects exceed its numerical interface")
                    section_authority = "registered prepared single numerical worker; no aggregation"
                # Reject unsupported physical preparation before sealing the
                # optional alternative. Actual coordinates are guarded afresh
                # at the original team boundary, never evaluated here.
                build_native_accesses(sections,
                    {root: "fort_native_" + str(index) for index, root in enumerate(sorted(numerical_roots))},
                    "fort_cn", parameters={root: binding.name for root, binding in operation.bindings.items()
                        if not binding.rank and binding.dtype == "integer"})
                self.cpu_native = operation
                self.cpu_native_sections = sections
                self.cpu_native_section_authority = section_authority
            except CompilationError as error:
                self.cpu_native_decline = str(error)

    def _validate(self):
        builder = self.builder
        region = builder._consume_numerical_region(builder.analysis, builder.entry.qualified, self.original_nodes)
        if region is None or region.prepared is not self.prepared or region.effects is not self.effect:
            raise CompilationError("combined numerical team lost its registered prepared/effect pair")
        from compiler.scopes.source_reads import validate_source_reads
        validate_source_reads(builder, self.calls[0], self.source_reads)
        if _source_partition(builder, self.original_nodes) != self.partition:
            raise CompilationError("combined numerical team original source partition changed")
        if self.cpu_native is not None:
            if self.cpu_native_section_authority == "registered prepared single numerical worker; no aggregation":
                if (len(self.prepared.offload.units) != 1 or len(self.effect.steps) != 1
                        or not self.prepared.all_runtime_guards_typed
                        or self.cpu_native_sections is not self.effect.steps[0].sections):
                    raise CompilationError("original CPU team lost exact single-worker physical section authority")
            elif (self.cpu_native_section_authority != "independent original native sections"
                    or self.cpu_native_sections.public() != self.cpu_native.coherence_sections.public()):
                raise CompilationError("original CPU team lost independent native section authority")
            operation = self.cpu_native
            analysis = operation.proof_analysis
            public = operation.summary["native_completion"]
            proof = analysis._joined_completions.get(public["proof_identity"])
            _graph, selected = analysis._selected_source(builder.entry.qualified, operation.nodes)
            if proof is None or proof.public() != public or selected != proof.selected_node_ids:
                raise CompilationError("original CPU team lost independent native completion authority")
            proof.validate(analysis, builder.entry.qualified, selected)

    def public(self):
        result = super().public()
        result["combined_numerical_team"] = {
            "schema_version": 2, "completion": self.completion.public(),
            "group_kind": self.partition["group_kind"], "unit_count": self.partition["unit_count"],
            "preparation_identity": self.prepared.identity, "effect_proof_identity": self.effect.identity,
            "source_reads": self.source_reads.public() if self.source_reads is not None else None,
            "original_span_sha256": self.partition["source_sha256"],
            "pristine_original_entry": {
                "source_eligible": not any(getattr(getattr(node, "item", None), "label", None)
                                           for original in self.original_nodes for node in walk(original)),
                "requires": "ordinary nonborrowed entry or immutable owning native latch; complete original team and source authority",
                "execution": "original whole team directly; no runtime acquisition, commit or copyback",
                "independent_native_effects_required": False,
            },
            "parallel_clauses": list(self.partition["parallel_clauses"]),
            "do_clauses": list(self.partition["do_clauses"]),
            "execution": "original team; optional installed-query native range before fork; other placements vote before master-local preparation",
            "fallback": "prequery or guard failure: publish and close before original worksharing body once",
            "cpu_choice": "consume the checked schedule with equivalence-proved collective CPU workers; retain context and definitions",
            "worker_resource_failure": "existing checked collective generated CPU worker once; coherent commit and wait",
            "post_execution_failure": "hard failure; no native or generated replay",
            "native_effects_authority": False, "closed_is_absorbing": True,
            "nested_team_created": False, "expected_omp_level": 1,
            "host_threads": self.builder.config.host_threads,
            "original_cpu_alternative": {
                "available": self.cpu_native is not None, "reason": self.cpu_native_decline,
                "authority": "independent original joined-native source, effect, storage and completion proofs",
                "physical_section_authority": self.cpu_native_section_authority,
                "numerical_effect_identity": (self.effect.identity if self.cpu_native_section_authority ==
                    "registered prepared single numerical worker; no aggregation" else None),
                "schedule": "atomic exact all-CPU current worker range; future GPU choices retained",
                "aliases": "duplicate canonical handles skip the optional native route; generated worker keeps its existing alias authority",
                "views": "original source base and shape must equal the registered root before native acquisition",
                "native_summary_identity": self.cpu_native.summary["summary_identity"] if self.cpu_native else None,
                "native_completion": self.cpu_native.summary["native_completion"] if self.cpu_native else None,
                "sections": self.cpu_native_sections.public() if self.cpu_native else None,
                "pristine_inherited_query": {
                    "available": self.cpu_native is not None and not any(
                        getattr(getattr(node, "item", None), "label", None)
                        for original in self.original_nodes for node in walk(original)),
                    "requires": "already installed query; reached guards; exact all-CPU current worker range",
                    "execution": "native acquisition before original fork; exact original source; host end after join",
                    "original_span_sha256": self.partition["source_sha256"],
                    "mixed_or_unavailable": "unchanged generated participant protocol",
                },
            },
        }
        if self.cpu_native is not None:
            result["combined_numerical_team"].update({
                "cpu_choice": "original Fortran body once under independent exact native hooks when current worker range is all CPU; generated workers otherwise",
                "native_effects_authority": True,
            })
        if hasattr(self, "borrowed_fallback"):
            fallback = self.borrowed_fallback
            result["combined_numerical_team"].update({
                "fallback": "independent complete native plan; validate and host begin before original body once; host end after join",
                "cpu_choice": "cancel untouched all-CPU numerical plan; independent original native effects retain context and definitions",
                "native_effects_authority": True, "closed_is_absorbing": False,
                "context": "borrowed and never closed by this worker",
                "native_summary_identity": fallback.summary["summary_identity"],
                "native_completion": fallback.summary["native_completion"],
                "native_required_whole": sorted(root for root, actions in fallback.coherence_effects.items()
                                                if "read" in actions or root not in fallback.overwrites),
                "whole_on_return": sorted(fallback.overwrites),
                "whole_return_authority": "same exact original group: independent source must-writes and validated numerical equivalence",
                "native_validation_failure": "checked hard failure; borrowed context cannot close or replay",
            })
        return result

    def _emit_cpu_native(self, handles, values, imports, specification, block):
        """Bind an independently proved original body to the current CPU range."""
        from compiler.emission.fortran.formatting import _fortran_list
        from compiler.scopes.access import build_native_accesses
        from compiler.scopes.provenance import position, source_id
        from compiler.scopes.source import _checked

        operation = self.cpu_native
        call, = self.calls
        public, _ = self.builder.entry_artifacts(call.procedure)
        entry = public["planning"].get("native_begin")
        if operation is None or not entry or not entry["available"]:
            return (), ()
        mapped = [handles[resource.resource] for resource in self.cpu_native_sections.resources]
        if len(set(mapped)) != len(mapped):
            return (), ()
        accesses = build_native_accesses(self.cpu_native_sections, handles, "fort_cn",
            parameters={root: values.get(root, self.builder.visible(self.builder.entry, root))
                        for root, binding in operation.bindings.items()
                        if not binding.rank and binding.dtype == "integer"},
            logical_lower_bounds={root: tuple(f"lbound({values[root]},{axis},kind=c_int64_t)"
                for axis in range(1, binding.rank + 1)) for root, binding in operation.bindings.items()
                if binding.rank and root in values}, on_error=("exit " + block,))
        specification += ["integer(c_int) :: fort_team_native_applied", "logical :: fort_team_try_native",
                          "logical :: fort_team_native_source",
                          f"type(fort_scope_plan_binding), target :: fort_team_native_bindings({max(1, len(accesses))})"]
        prepare = ["fort_team_native_source = .true."]
        for index, access in enumerate(accesses, 1):
            specification.extend(access.specification)
            prepare += [*access.prepare, f"fort_team_native_bindings({index}) = fort_scope_plan_binding()",
                        f"fort_team_native_bindings({index})%buffer = {access.handle}",
                        f"fort_team_native_bindings({index})%access = {access.access_name}"]
            resource = self.cpu_native_sections.resources[index-1]
            original = self.builder.visible(self.builder.entry, resource.resource)
            prefix = "fort_cn_" + str(index-1)
            prepare += [f"if (.not. is_contiguous({original})) fort_team_native_source = .false.",
                        f"if (any(shape({original},kind=c_size_t) /= {prefix}_extents)) &",
                        "fort_team_native_source = .false.",
                        f"if (all(shape({original},kind=c_size_t) > 0_c_size_t)) then",
                        f"if (.not. c_associated(c_loc({original}), {prefix}_layout%host)) &",
                        "fort_team_native_source = .false.", "endif"]
        parameters = {parameter.name.lower(): parameter for parameter in call.region.parameters}
        arguments = ["fort_context"]
        for array in public["array_parameters"]:
            arguments.append(handles[parameters[array["name"].lower()].resource])
        for scalar in public["scalar_parameters"]:
            if scalar["name"] not in public["planning"]["scalar_inputs"]:
                continue
            parameter = parameters[scalar["name"].lower()]
            if parameter.lower_bound_dimension is not None:
                arguments.append(f"int(lbound({values[parameter.resource]},{parameter.lower_bound_dimension},kind=c_int64_t),c_int)")
            else:
                arguments.append(values.get(parameter.resource, self.builder.visible(self.builder.entry, parameter.resource)))
        arguments += ["c_loc(fort_team_native_bindings)" if accesses else "c_null_ptr",
                      str(len(accesses)) + "_c_size_t", "fort_team_native_applied"]
        imports.append(f"use {public['fortran_module']}, only: fort_team_native_begin => {entry['fortran_procedure']}")
        prepare += ["if (fort_team_native_source) then", "fort_team_native_applied = 0_c_int",
                    *_fortran_list("fort_status = fort_team_native_begin(", arguments, ")", 0),
                    "if (fort_status == FORT_SCOPE_OK .and. fort_team_native_applied /= 0_c_int) then",
                    "fort_team_native = .true.",
                    *position(self.builder, implementation=source_id(self.builder, "implementation",
                        span=(self.first, self.last), backend="original_native")), "endif", "endif"]
        distinct = [f"{first} /= {second}" for index, first in enumerate(mapped) for second in mapped[index+1:]]
        if distinct:
            prepare = ["if (" + " .and. ".join(distinct) + ") then", *prepare, "endif"]
        ends = [line for access in accesses for line in _checked(f"fort_scope_host_end(fort_context, {access.handle})")]
        return prepare, ends

    def emit_original(self, owner, unit, handles, values, imports, specification, *, preparation, block, price_units):
        """Prepare inside the actual team; never exit an enclosing OpenMP block."""
        self._validate()
        close_owner = copy(owner)
        close_owner.context, close_owner.enabled, close_owner.control_guard = (
            "fort_team_context", "fort_team_enabled", None)
        graph_epoch = self.builder.config.scope_execution == "graph"
        if graph_epoch:
            close_owner.epoch_pristine = "fort_team_epoch_pristine"
            close_owner.query_plan = "fort_team_query_plan"
        failure = [*close_owner.close(prework=True, prework_status="fort_status"), "fort_native_ready = .false."]
        extra = ["logical :: fort_team_run, fort_team_native, fort_team_owned, fort_team_enabled",
                 "logical :: fort_team_environment_supported", "integer(c_int64_t) :: fort_team_context",
                 *[f"integer(c_int64_t) :: {handle}" for handle in handles.values()]]
        before = [f"fort_team_context = {owner.context}", f"fort_team_enabled = {owner.enabled}",
                  *[f"{handle} = fort_buffer_{owner.resources[root][1]}" for root, handle in handles.items()],
                  "fort_team_owned = .true."]
        after = ["if (fort_team_owned) then", f"{owner.context} = fort_team_context",
                 f"{owner.enabled} = fort_team_enabled",
                 *[f"fort_buffer_{owner.resources[root][1]} = {handle}" for root, handle in handles.items()], "endif"]
        if graph_epoch:
            extra += ["logical :: fort_team_epoch_pristine, fort_team_query_plan"]
            before += [f"fort_team_epoch_pristine = {owner.epoch_pristine}",
                       f"fort_team_query_plan = {owner.query_plan}"]
            after[-1:-1] = [f"{owner.epoch_pristine} = fort_team_epoch_pristine",
                            f"{owner.query_plan} = fort_team_query_plan"]
        all_specification = [*specification, *extra]
        lines, generated = self._emit_team(handles, values, imports, all_specification,
            preparation=preparation, block=block, failure=failure, price_units=price_units,
            owner=owner, unit=unit)
        extra += all_specification[len(specification) + len(extra):]
        _narrow_runtime_imports(imports, [*specification, *extra], [*before, *generated, *after])
        _check_generated_names(self.original_nodes, [*specification, *extra], imports, generated, block)
        replacement = "\n".join(fortran_lines(lines)) + "\n"
        return extra, [(self.first, self.last, replacement)], before, after

    def emit_borrowed(self, handles, values, imports, specification, *, block):
        """Keep the borrowed context on a separately proved whole native path."""
        from compiler.scopes.numerical_preflight import preparation_preflight
        from compiler.scopes.source import _checked

        self._validate()
        if self.source_reads is not None:
            raise CompilationError("original read admission requires an original lexical owning team")
        operation = self.borrowed_fallback
        if (len(self.native) != 1 or self.native[0] is not operation
                or len(self.guarded) != 1 or self.guarded.get(id(self.calls[0].node)) is not operation
                or self.close_guards or len(operation.nodes) != len(self.original_nodes)
                or any(left is not right for left, right in zip(operation.nodes, self.original_nodes, strict=True))):
            raise CompilationError("borrowed numerical team lost its exact independent native fallback")
        guard_specification, preparation = preparation_preflight(self.builder, self.calls[0], values)
        extra = ["logical :: fort_team_run, fort_team_native, fort_team_enabled, fort_native_ready",
                 "logical :: fort_team_environment_supported", "integer(c_int64_t) :: fort_team_context",
                 "logical :: fort_numerical_guard", *guard_specification]
        preparation += ["if (.not. fort_numerical_guard) then", "fort_native_ready = .false.",
                        "exit " + block, "endif"]
        # The fallback is a complete independent source effect proof. Rebuild
        # its own conservative plan after any failed/declined numerical query;
        # no residual numerical binding or status can authorize host access.
        native_begin = _checked("fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)")
        for index, (root, actions) in enumerate(sorted(operation.coherence_effects.items()), 1):
            flags = (["FORT_SCOPE_READ_ALL"] if "read" in actions else [])
            flags += ["FORT_SCOPE_WRITE_ALL"] if "write" in actions else []
            flags += ["FORT_SCOPE_OVERWRITE_ALL"] if root in operation.overwrites else []
            native_begin += [f"fort_bindings({index}) = fort_scope_plan_binding()",
                             f"fort_bindings({index})%buffer = {handles[root]}",
                             f"fort_bindings({index})%access%flags = " + " + ".join(flags)]
        count = len(operation.coherence_effects)
        native_begin += ["fort_status = fort_scope_plan_add( &",
            "    fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &",
            f"    {'c_loc(fort_bindings)' if count else 'c_null_ptr'}, {count}_c_size_t, &",
            "    0.0_c_double, 0.0_c_double, 0_c_int)",
            "if (fort_status /= FORT_SCOPE_OK) error stop 'borrowed native team plan failed'"]
        native_begin += _checked("fort_scope_plan_validate(fort_context)")
        from compiler.scopes.access import native_access_begin
        native_begin += _checked(native_access_begin(count))
        native_end = [line for root in sorted(operation.coherence_effects)
                      for line in _checked(f"fort_scope_host_end(fort_context, {handles[root]})")]
        if self.builder.config.scope_execution != "graph":
            native_end += _checked("fort_scope_wait(fort_context)")
        before = ["fort_team_context = fort_context", "fort_team_enabled = .true.",
                  "fort_native_ready = fort_mode /= 0_c_int"]
        lines, generated = self._emit_team(handles, values, imports, [*specification, *extra], preparation=preparation,
            block=block, failure=["fort_native_ready = .false."], native_begin=native_begin,
            native_end=native_end, price_units=self.builder.config.policy == "auto")
        # The caller adds entry definition events before this team. Finalize
        # imports and names only after that complete worker has been assembled.
        return extra, [*before, *lines], [*before, *generated]

    def _emit_team(self, handles, values, imports, specification, *, preparation, block, failure, price_units,
                   owner=None, unit=None,
                   native_begin=(), native_end=()):
        """One original participant protocol for owning and borrowed workers."""
        builder = self.builder
        call, = self.calls
        inherited = owner is not None and builder.config.scope_execution == "graph"
        imports.append("use omp_lib, only: fort_team_omp_level => omp_get_level, fort_team_omp_threads => omp_get_num_threads")
        from compiler.scopes.source_reads import build_source_reads
        read_code = build_source_reads(builder, self.source_reads, handles, values, "fort_source_read",
            call=call, on_error=(*failure, "exit " + block),
            logical_lower_bounds={root: tuple(f"lbound({values[root]}, {axis}, kind=c_int64_t)"
                for axis in range(1, binding.rank + 1)) for root, binding in
                ((binding.root, binding) for binding in call.region.bindings if binding.rank)})
        specification.extend(read_code.specification)
        cpu_begin, cpu_end = (self._emit_cpu_native(handles, values, imports, specification, block)
                              if owner is not None and price_units and self.cpu_native is not None else ((), ()))
        # A retained query has already installed this exact worker sequence.
        # Try only its current independently proved native range before the
        # original fork. Mixed ranges retain the participant protocol below.
        # Source labels have procedure scope and cannot be duplicated into
        # mutually exclusive original/generated branches.
        pristine = None
        unmanaged = ordinary_entry_guard(owner) if owner is not None else None
        if inherited and (cpu_begin or unmanaged is not None):
            pristine = original_team_source(builder, self.original_nodes, self.first, self.last,
                                            self.partition["source_sha256"])
        if pristine is not None:
            specification += ["logical :: fort_team_pristine"]
        shared = tuple(dict.fromkeys((*declared_names(specification), "fort_native_ready")))
        parallel = "!$omp parallel" + "".join(" " + clause for clause in self.partition["parallel_clauses"])
        parallel += " shared(" + ",".join(shared) + ")"
        do = "!$omp do" + "".join(" " + clause for clause in self.partition["do_clauses"])
        lines = [parallel, "!$omp master", "fort_team_run = .false.", "fort_team_native = .false.",
                 "fort_team_environment_supported = .true.", "if (fort_native_ready) then",
                 f"if (fort_team_omp_level() /= 1 .or. fort_team_omp_threads() /= {builder.config.host_threads}) then",
                 *failure, "endif", "endif", "!$omp end master", "!$omp barrier",
                 "if (fort_native_ready) then", "if (fort_scope_numerical_environment_supported() == 0) then",
                 "!$omp atomic write", "fort_team_environment_supported = .false.",
                 "endif", "endif", "!$omp barrier", "!$omp master", "if (fort_native_ready) then",
                 "associate(fort_context => fort_team_context)", block + ": block",
                 "if (.not. fort_team_environment_supported) then", *failure, "exit " + block, "endif",
                 *preparation,
                 *(["if (.not. fort_team_query_plan) then"] if inherited else []),
                 *read_code.prepare,
                 "fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)",
                 "if (fort_status == FORT_SCOPE_OK) then",
                 *builder.inline.emit_call(call, handles, values, imports, query=True),
                 "endif",
                 "if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)"]
        if owner is not None and owner.interval_placement:
            from compiler.offload.selection_cost import choice
            lines += ["if (fort_status == FORT_SCOPE_OK) then", *choice(owner, unit, handles, values), "endif",
                      "if (.not. fort_native_ready) exit " + block]
        if price_units:
            public, _ = builder.entry_artifacts(call.procedure)
            imports.append(f"use {public['fortran_module']}, only: fort_team_choose => {public['planning']['fortran_selector']}")
            lines += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_team_choose(fort_context, fort_decision)"]
            if owner is None:
                # This borrowed route has an independent original-native
                # effect proof and rebuilds that plan below. Relinquish only
                # the untouched all-CPU numerical schedule before doing so.
                # Owning teams instead consume its CPU choices through the
                # existing collective worker and canonical host accesses.
                lines += ["if (fort_status == FORT_SCOPE_OK .and. fort_decision%gpu_units == 0) then",
                          "fort_status = fort_scope_plan_cancel_native_v1(fort_context)",
                          "if (fort_status /= FORT_SCOPE_OK) error stop 'borrowed native choice cancellation failed'",
                          *failure, "exit " + block, "endif"]
        if inherited:
            lines += ["endif"]
        if cpu_begin:
            lines += ["fort_team_try_native = .false."]
            if inherited:
                lines += ["if (fort_team_query_plan) then", "fort_team_try_native = .true.", "else"]
            lines += ["fort_team_try_native = fort_decision%gpu_units == 0"]
            if inherited:
                lines += ["endif"]
            lines += ["if (fort_status == FORT_SCOPE_OK .and. fort_team_try_native) then", *cpu_begin, "endif"]
        native_body = self.partition["body"].splitlines()
        lines += ["if (fort_status /= FORT_SCOPE_OK) then", *failure, "exit " + block, "endif",
                  "fort_team_run = .not. fort_team_native", "end block " + block, "end associate",
                  # Any preparation-only exit, including a cheap native choice,
                  # must finish native coherence before the original body.
                  "if (.not. fort_team_run .and. .not. fort_team_native) then", *failure, "endif", "endif",
                  *(["if (.not. fort_team_run) then", *native_begin, "endif"] if native_begin else []),
                  "!$omp end master",
                  "!$omp barrier", "if (fort_team_run) then", "block", "integer(c_int) :: fort_returned",
                  "associate(fort_context => fort_team_context)",
                  *builder.inline.emit_call(call, handles, values, imports, query=False, collective=True,
                                            status="fort_returned", check_status=False),
                  "!$omp master", "fort_status = fort_returned"]
        if inherited:
            lines += ["if (fort_status /= FORT_SCOPE_OK) then",
                "if (.not. fort_team_query_plan) error stop 'combined original team numerical execution failed'",
                "if (fort_status /= FORT_SCOPE_ARGUMENT .and. fort_status /= FORT_SCOPE_BOUNDARY .and. &",
                "fort_status /= FORT_SCOPE_UNINITIALIZED .and. fort_status /= FORT_SCOPE_PLAN_MISMATCH) &",
                "error stop 'inherited collective entry executed or invalid state'",
                *failure, "fort_team_run = .false.", "endif"]
        else:
            lines += ["if (fort_status /= FORT_SCOPE_OK) error stop 'combined original team numerical execution failed'"]
        lines += [
                  *([] if self.builder.config.scope_execution == "graph" else [
                      "fort_status = fort_scope_wait(fort_context)",
                      "if (fort_status /= FORT_SCOPE_OK) error stop 'combined original team completion failed'"]),
                  "!$omp end master", "!$omp barrier", "end associate", "end block",
                  "endif", "if (.not. fort_team_run) then"]
        native = ([do, *native_body, "!$omp end do"]
                  if self.partition["group_kind"] == "combined_parallel_do" else native_body)
        generated = list(lines)
        lines += native
        if cpu_end:
            finishing = ["!$omp barrier", "!$omp master", "if (fort_team_native) then",
                         "associate(fort_context => fort_team_context)", *cpu_end, "end associate", "endif",
                         "!$omp end master", "!$omp barrier"]
            lines += finishing
            generated += finishing
        if native_end:
            # A separate group's last DO may have NOWAIT. The explicit team
            # join must precede publication of any original native writes.
            finishing = ["!$omp barrier", "!$omp master", *native_end, "!$omp end master", "!$omp barrier"]
            lines += finishing
            generated += finishing
        suffix = ["endif", "!$omp end parallel"]
        lines, generated = [*lines, *suffix], [*generated, *suffix]
        if pristine is not None:
            preteam = ["fort_team_pristine = .false.", "fort_team_native = .false."]
            if unmanaged is not None:
                preteam += [f"if ({unmanaged}) fort_team_pristine = .true."]
            if cpu_begin:
                preteam += ["if (.not. fort_team_pristine) then", "if (fort_native_ready) then",
                            "if (fort_team_query_plan) then", "associate(fort_context => fort_team_context)",
                            block + ": block", *preparation, *cpu_begin,
                            "if (fort_status /= FORT_SCOPE_OK) then", *failure, "exit " + block, "endif",
                            "fort_team_pristine = fort_team_native",
                            "end block " + block, "end associate", "endif", "endif", "endif"]
            ending = (["if (fort_team_native) then", "associate(fort_context => fort_team_context)",
                       *cpu_end, "end associate", "endif"] if cpu_end else [])
            lines = [*preteam, "if (fort_team_pristine) then", *pristine.splitlines(),
                     *ending, "else", *lines, "endif"]
            generated = [*preteam, "if (fort_team_pristine) then", *ending, "else", *generated, "endif"]
        return lines, generated
