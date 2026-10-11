"""One original combined numerical team with a whole-group native exit.

Numerical completion never supplies native memory hooks. Every original team
member votes before master's preparation, and a rejected unit closes before
the preserved worksharing body executes once.
"""

import re
from copy import copy
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.native_group_structure import NATIVE_GROUP_SCAN_LIMIT
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.joined_team import declared_names
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
    count = 0
    for original in originals:
        for node in walk(original):
            count += 1
            if count > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("combined team routing scan budget exhausted")
            text = directive(node)
            if text is not None:
                return re.match(r"parallel\s+do(?:\s|$)", text) is not None
    return False


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
    if not directives or not re.match(r"parallel\s+do(?:\s|$)", directives[0][1]):
        raise CompilationError("combined numerical team requires one original PARALLEL DO")
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
    if not first < body_first <= body_last <= last:
        raise CompilationError("combined numerical team source association is unordered")
    source = builder.entry.scope.path
    if source.stat().st_size > 32 * 1024**2:
        raise CompilationError("combined numerical team original source byte budget exhausted")
    lines = source.read_text().splitlines(keepends=True)
    raw = "".join(lines[first-1:last])
    if (last > len(lines) or len(raw.encode()) > NATIVE_GROUP_SCAN_LIMIT
            or any(line.lstrip().startswith("#") for line in lines[first-1:last])):
        raise CompilationError("combined numerical team requires a bounded original mapped source span")
    boundaries = (opening, header, end, *((ending,) if ending is not None else ()))
    if any(";" in line for node in boundaries for line in lines[statement_span(node)[0]-1:statement_span(node)[1]]):
        raise CompilationError("combined numerical team cannot replace a semicolon source boundary")
    parallel, worksharing = [], []
    remainder = re.sub(r"^parallel\s+do\b", "", text, count=1).strip()
    clauses = 0
    while remainder:
        clauses += 1
        if clauses > builder.analysis.operation_limit:
            raise CompilationError("combined numerical team clause budget exhausted")
        match = re.match(r"(private|shared|default|collapse|schedule)\s*\(([^()]*)\)\s*", remainder)
        if match is None:
            raise CompilationError("combined numerical team has an unsupported clause partition: " + remainder)
        kind, value = match.group(1), match.group(2).strip()
        # Completion has authenticated these original operands and bindings.
        # Repeat the finite shape checks so no new clause semantics is implied.
        if ((kind == "default" and value not in {"none", "shared"})
                or (kind == "collapse" and value not in {"1", "2", "3", "4"})
                or (kind == "schedule" and value not in {"static", "runtime"})
                or (kind in {"private", "shared"} and any(
                    not re.fullmatch(r"[a-z][a-z0-9_]*", name.strip()) for name in value.split(",")))):
            raise CompilationError("combined numerical team clause operands are unsupported")
        clause = kind + "(" + value + ")"
        (worksharing if kind in {"collapse", "schedule"} else parallel).append(clause)
        remainder = remainder[match.end():].lstrip(", ")
    return {"first": first, "last": last, "body": "".join(lines[body_first-1:body_last]),
            "source_sha256": sha256(raw.encode()).hexdigest(),
            "parallel_clauses": tuple(parallel), "do_clauses": tuple(worksharing),
            "scan_nodes": count, "clause_count": clauses}


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


def _narrow_runtime_imports(imports, specification, generated):
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
    used = set(re.findall(r"\b[a-z][a-z0-9_]*\b", "\n".join((*specification, *generated)).lower()))
    for index, statement in enumerate(imports):
        lower = statement.strip().lower()
        if lower in {"use fort_scoped_memory", "use iso_c_binding"}:
            selected = sorted(used & (exports if lower.endswith("fort_scoped_memory") else iso))
            if not selected:
                raise CompilationError("combined team generated runtime import has no required exports")
            imports[index] = statement + ", only: " + ", ".join(selected)


class CombinedNumericalTeam(StructuredScope):
    """A single exact prepared group, without an active native alternative."""

    def __init__(self, builder, nodes):
        self.builder, self.nodes = builder, tuple(nodes)
        self.original_nodes = original_roots(builder.inline.original_selection(nodes))
        self.prepared = builder.numerical_preparation_for(
            builder.analysis, builder.entry.qualified, self.original_nodes)
        if self.prepared is None:
            raise CompilationError("combined numerical team requires registered original preparation")
        self.prepared.validate(builder.analysis, builder.entry.qualified, self.original_nodes)
        self.prepared.validate_configuration(options=builder.options, config=builder.config,
            environment_root=builder._numerical_environment_root,
            immutable_inputs=getattr(builder, "immutable_inputs", {}))
        if not self.prepared.completion.get("has_openmp_in_closure"):
            raise CompilationError("combined numerical team requires original joined completion")
        self.partition = _source_partition(builder, self.original_nodes)
        self.first, self.last = self.partition["first"], self.partition["last"]
        facade = builder.inline.prepare_combined_team(self.original_nodes)
        call = builder.inline.call(facade)
        self.effect = builder.call_numerical_effect(call)
        if self.effect is None or self.effect.prepared is not self.prepared:
            raise CompilationError("combined numerical team requires exact registered physical effects")
        self.effect.validate(builder.analysis, builder.entry.qualified, self.original_nodes, self.prepared)
        self.completion = self.prepared.completion_proof
        self.native_metadata = True
        self.native_host_reads = False
        self.calls, self.native, self.segments = [call], [], []
        self.guarded = {}
        self.close_guards = {id(call.node): self.effect}
        self.tree = [Segment([call])]
        self.operation_count = self.effect.public()["ordered_effect_count"]
        if self.operation_count > builder.analysis.operation_limit:
            raise CompilationError("combined numerical team ordered effect budget exhausted")

    def _validate(self):
        builder = self.builder
        self.prepared.validate(builder.analysis, builder.entry.qualified, self.original_nodes)
        self.prepared.validate_configuration(options=builder.options, config=builder.config,
            environment_root=builder._numerical_environment_root,
            immutable_inputs=getattr(builder, "immutable_inputs", {}))
        self.effect.validate(builder.analysis, builder.entry.qualified, self.original_nodes, self.prepared)
        if _source_partition(builder, self.original_nodes) != self.partition:
            raise CompilationError("combined numerical team original source partition changed")

    def public(self):
        result = super().public()
        result["combined_numerical_team"] = {
            "schema_version": 1, "completion": self.completion.public(),
            "preparation_identity": self.prepared.identity, "effect_proof_identity": self.effect.identity,
            "original_span_sha256": self.partition["source_sha256"],
            "parallel_clauses": list(self.partition["parallel_clauses"]),
            "do_clauses": list(self.partition["do_clauses"]),
            "execution": "original team; all-participant environment vote; master-local preparation",
            "fallback": "prequery, guard or native choice: publish and close before original worksharing body once",
            "worker_resource_failure": "existing checked collective generated CPU worker once; coherent commit and wait",
            "post_execution_failure": "hard failure; no native or generated replay",
            "native_effects_authority": False, "closed_is_absorbing": True,
            "nested_team_created": False, "expected_omp_level": 1,
            "host_threads": self.builder.config.host_threads,
        }
        return result

    def emit_original(self, owner, unit, handles, values, imports, specification, *, preparation, block):
        """Prepare inside the actual team; never exit an enclosing OpenMP block."""
        self._validate()
        builder = self.builder
        call, = self.calls
        imports.append("use omp_lib, only: fort_team_omp_level => omp_get_level, fort_team_omp_threads => omp_get_num_threads")
        close_owner = copy(owner)
        close_owner.context, close_owner.enabled, close_owner.control_guard = (
            "fort_team_context", "fort_team_enabled", None)
        failure = [*close_owner.close(), "fort_native_ready = .false."]
        extra = ["logical :: fort_team_run, fort_team_native, fort_team_owned, fort_team_enabled",
                 "logical :: fort_team_environment_supported", "integer(c_int64_t) :: fort_team_context",
                 *[f"integer(c_int64_t) :: {handle}" for handle in handles.values()]]
        before = [f"fort_team_context = {owner.context}", f"fort_team_enabled = {owner.enabled}",
                  *[f"{handle} = fort_buffer_{owner.resources[root][1]}" for root, handle in handles.items()],
                  "fort_team_owned = .true."]
        after = ["if (fort_team_owned) then", f"{owner.context} = fort_team_context",
                 f"{owner.enabled} = fort_team_enabled",
                 *[f"fort_buffer_{owner.resources[root][1]} = {handle}" for root, handle in handles.items()], "endif"]
        shared = tuple(dict.fromkeys((*declared_names([*specification, *extra]), "fort_native_ready")))
        parallel = "!$omp parallel" + "".join(" " + clause for clause in self.partition["parallel_clauses"])
        parallel += " shared(" + ",".join(shared) + ")"
        do = "!$omp do" + "".join(" " + clause for clause in self.partition["do_clauses"])
        lines = [parallel, "!$omp master", "fort_team_run = .false.",
                 "fort_team_environment_supported = .true.", "if (fort_native_ready) then",
                 f"if (fort_team_omp_level() /= 1 .or. fort_team_omp_threads() /= {builder.config.host_threads}) then",
                 *failure, "endif", "endif", "!$omp end master", "!$omp barrier",
                 "if (fort_native_ready) then", "if (fort_scope_numerical_environment_supported() == 0) then",
                 "!$omp atomic write", "fort_team_environment_supported = .false.",
                 "endif", "endif", "!$omp barrier", "!$omp master", "if (fort_native_ready) then",
                 "associate(fort_context => fort_team_context)", block + ": block",
                 "if (.not. fort_team_environment_supported) then", *failure, "exit " + block, "endif",
                 *preparation,
                 "fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)",
                 "if (fort_status == FORT_SCOPE_OK) then",
                 *builder.inline.emit_call(call, handles, values, imports, query=True),
                 "endif",
                 "if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)"]
        if builder.config.policy == "auto":
            public, _ = builder.entry_artifacts(call.procedure)
            imports.append(f"use {public['fortran_module']}, only: fort_team_choose => {public['planning']['fortran_selector']}")
            lines += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_team_choose(fort_context, fort_decision)",
                      "if (fort_status == FORT_SCOPE_OK .and. fort_decision%gpu_units == 0) then",
                      *failure, "exit " + block, "endif"]
        native_body = self.partition["body"].splitlines()
        lines += ["if (fort_status /= FORT_SCOPE_OK) then", *failure, "exit " + block, "endif",
                  "fort_team_run = .true.", "end block " + block, "end associate",
                  # Any preparation-only exit, including a cheap native choice,
                  # must publish before the original body can touch host data.
                  "if (.not. fort_team_run) then", *failure, "endif", "endif", "!$omp end master",
                  "!$omp barrier", "if (fort_team_run) then", "block", "integer(c_int) :: fort_returned",
                  "associate(fort_context => fort_team_context)",
                  *builder.inline.emit_call(call, handles, values, imports, query=False, collective=True,
                                            status="fort_returned", check_status=False),
                  "!$omp master", "fort_status = fort_returned",
                  "if (fort_status /= FORT_SCOPE_OK) error stop 'combined original team numerical execution failed'",
                  "fort_status = fort_scope_wait(fort_context)",
                  "if (fort_status /= FORT_SCOPE_OK) error stop 'combined original team completion failed'",
                  "!$omp end master", "!$omp barrier", "end associate", "end block", "else", do,
                  *native_body, "!$omp end do", "endif", "!$omp end parallel"]
        generated = [*lines[:len(lines)-len(native_body)-3], *lines[-3:]]
        _narrow_runtime_imports(imports, [*specification, *extra], [*before, *generated, *after])
        _check_generated_names(self.original_nodes, [*specification, *extra], imports, generated, block)
        replacement = "\n".join(fortran_lines(lines)) + "\n"
        return extra, [(self.first, self.last, replacement)], before, after
