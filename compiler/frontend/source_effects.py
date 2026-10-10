"""Bounded source effects for native calls, independent of numerical lowering.

This analysis never executes bounds or rewrites original calls. Its sections are
conservative whole resources; logical array subscripts are retained as evidence,
not mistaken for physical transfer coordinates. Unknown effects are boundaries.
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from copy import copy, deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.lowering import _KindScope
from compiler.frontend.native_sections import analyze_native_sections
from compiler.frontend.source_inputs import SourceInputs
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

# Bump when source-effect, call-composition or summary semantics change.
SOURCE_SUMMARY_VERSION = 14
_DEFAULT_SUMMARY_CACHE = SummaryCache()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _part(node, name):
    return next((child for child in _children(node) if _kind(child) == name), None)


def _attach_execution_directives(node):
    """Recover original executable prefixes that fparser stores in specs.

    Move the existing comment nodes, never manufactured spans or directives.
    Only a trailing comment suffix beginning a parallel region qualifies;
    declaration directives and comments separated by declarations stay put.
    All source identities are captured after this parser normalization.
    """
    specification, execution = _part(node, "Specification_Part"), _part(node, "Execution_Part")
    def detach_prefixes(parent):
        if not hasattr(parent, 'content'):
            return
        children = []
        for child in parent.content:
            if _kind(child) in {'Block_Nonlabel_Do_Construct', 'Block_Label_Do_Construct',
                               'If_Construct', 'Associate_Construct'}:
                while child.content and _kind(child.content[0]) == 'Comment':
                    children.append(child.content.pop(0))
            detach_prefixes(child)
            children.append(child)
        parent.content[:] = children

    # A previous team's join must not become part of a following loop merely
    # because fparser attached its leading comments there. Preserve the actual
    # comment objects and their order before recording original-source roles.
    if execution is not None:
        detach_prefixes(execution)
    if specification is None or execution is None:
        return
    containers, trailing = [], []
    for child in reversed(specification.content):
        if _kind(child) == "Comment":
            containers.append((specification, child))
            trailing.insert(0, child)
        elif _kind(child) == "Implicit_Part":
            for item in reversed(child.content):
                if _kind(item) != "Comment":
                    break
                containers.append((child, item))
                trailing.insert(0, item)
            if any(_kind(item) != "Comment" for item in child.content):
                break
        else:
            break
    directives = [str(item).strip().lower() for item in trailing
                  if str(item).lstrip().lower().startswith("!$omp")]
    if not directives or re.match(r"!\$omp\s+parallel(?:\s|$)", directives[0]) is None:
        return
    for parent, comment in containers:
        del parent.content[next(index for index, item in enumerate(parent.content) if item is comment)]
    specification.content[:] = [child for child in specification.content
                                 if _kind(child) != "Implicit_Part" or child.content]
    execution.content[:0] = trailing


def _admit_cached_summary(item, routine, limit):
    """Validate cached public structures before source consumers use them."""
    def require(condition):
        if not condition:
            raise ValueError("cached source summary structure is invalid")

    def strings(values):
        return type(values) is list and all(type(value) is str for value in values)

    def binding(value):
        require(type(value) is dict and {"name", "resource", "type", "rank", "kind", "intent", "attributes", "lower_bounds", "shape"} <= value.keys())
        require(all(type(value.get(key)) is str for key in ("name", "resource", "type")))
        require(type(value.get("rank")) is int and value["rank"] >= 0)
        require(value.get("kind") is None or type(value["kind"]) is int)
        require(value.get("intent") is None or type(value["intent"]) is str)
        require(strings(value.get("attributes")) and strings(value.get("lower_bounds")))
        require(len(value["lower_bounds"]) == value["rank"])
        require(type(value.get("shape")) is list and len(value["shape"]) == value["rank"])
        for axis in value["shape"]:
            require(type(axis) is dict and type(axis.get("kind")) is str and type(axis.get("source")) is str)
            require(type(axis.get("bounds")) is list and all(bound is None or type(bound) is str for bound in axis["bounds"]))

    def bound(value):
        require(type(value) is dict and type(value.get("kind")) is str and type(value.get("expression")) is str)
        require(type(value.get("dependencies")) is list)
        for dependency in value["dependencies"]:
            require(type(dependency) is dict and type(dependency.get("resource")) is str
                    and dependency.get("kind") in {"scalar_read", "descriptor_read"})
        if "children" in value:
            require(type(value["children"]) is list)
            for child in value["children"]:
                bound(child)

    def mapping(value):
        require(type(value) is dict and {"formal", "formal_resource", "actual", "resource", "storage", "presence", "formal_descriptor", "actual_descriptor", "requirements", "section"} <= value.keys())
        require(type(value.get("formal")) is str and value.get("formal_resource") == "argument::" + value["formal"])
        require(value.get("actual") is None or type(value["actual"]) is str)
        require(value.get("resource") is None or type(value["resource"]) is str)
        require(value.get("storage") in {"whole", "rectangle", "expression", "omitted"})
        require(value.get("presence") in {"supplied", "omitted", "forwarded_optional", "allocation_dependent", "association_dependent"})
        binding(value["formal_descriptor"])
        if value["actual_descriptor"] is not None:
            binding(value["actual_descriptor"])
        require(type(value.get("requirements")) is dict and all(type(flag) is bool for flag in value["requirements"].values()))
        section = value["section"]
        require(section is None or type(section) is dict)
        if section is not None:
            require(type(section.get("resource")) is str and type(section.get("rank")) is int)
            require(type(section.get("axes")) is list and len(section["axes"]) == section["rank"])
            require(type(section.get("dependencies")) is list)
            for axis in section["axes"]:
                require(type(axis) is dict and axis.get("kind") == "unit_stride_range" and axis.get("stride") == 1)
                for axis_bound in (axis["lower"], axis["upper"]):
                    bound(axis_bound)

    require(type(item) is dict and item.get("procedure") == routine.qualified and item.get("complete") is True)
    require(item.get("source_kind") == routine.source_kind and item.get("reasons") == [])
    require(type(item.get("cloneable")) is bool)
    for key in ("definition_changes", "guaranteed_whole_overwrites", "persistent_state", "openmp_directives"):
        require(strings(item.get(key)))
    for key in ("arguments", "descriptor_requirements"):
        require(type(item.get(key)) is list)
        for value in item[key]:
            binding(value)
    require(type(item.get("definition_diagnostics")) is list)
    for diagnostic in item["definition_diagnostics"]:
        require(type(diagnostic) is dict and all(type(diagnostic.get(key)) is str
                                               for key in ("procedure", "resource", "source_access", "reason")))
    require(type(item.get("capture_lifetime_requirements")) is list)
    for requirement in item["capture_lifetime_requirements"]:
        require(type(requirement) is dict and type(requirement.get("resource")) is str
                and type(requirement.get("authorized")) is bool)
    completion, sections, composition = item["native_completion"], item["native_sections"], item["effect_composition"]
    require(type(completion) is dict and type(completion.get("available")) is bool)
    require("reason" in completion and (type(completion["reason"]) is str or completion["reason"] is None))
    require(type(completion.get("caller_contract")) is str)
    require(all(type(completion.get(key)) is bool for key in
                ("requires_serial_caller", "has_openmp_in_closure", "has_opaque_calls_in_closure")))
    require(type(sections) is dict and type(sections.get("available")) is bool and type(sections.get("resources")) is list)
    require(type(composition) is dict and composition.get("available") is True and composition.get("reason", False) is None
            and type(composition.get("coordinate_system")) is str)
    require(type(item.get("ordered_effects")) is list and len(item["ordered_effects"]) <= limit)
    require(type(composition.get("operations")) is int and composition["operations"] == len(item["ordered_effects"]))
    effects = {"read", "write", "overwrite", "descriptor_read"}
    for effect in item["ordered_effects"]:
        require(type(effect) is dict and effect.get("kind") in effects | {"definition_change"})
        require(type(effect.get("resource")) is str and type(effect.get("source_procedure")) is str)
        require(type(effect.get("view_chain")) is list and type(effect.get("guard_frames")) is list)
        for view in effect["view_chain"]:
            mapping(view)
        for guard in effect["guard_frames"]:
            require(type(guard) is dict and type(guard.get("procedure")) is str and type(guard.get("condition")) is str)
    require(type(item.get("operations")) is list and len(item["operations"]) <= limit)
    for operation in item["operations"]:
        require(type(operation) is dict and operation.get("kind") in effects | {"call", "native_contract", "control"})
        require(strings(operation.get("guard")))
        if operation["kind"] in effects:
            require(type(operation.get("resource")) is str and type(operation.get("rank")) is int and operation["rank"] >= 0)
            require(operation.get("section") == "whole" and type(operation.get("source_access")) is str)
        elif operation["kind"] == "call":
            require(type(operation.get("procedure")) is str and operation.get("complete") is True)
            require(type(operation.get("actual_arguments")) is list
                    and all(actual is None or type(actual) is str for actual in operation["actual_arguments"]))
            require(type(operation.get("resource_mapping")) is dict
                    and all(type(formal) is str and type(root) is str for formal, root in operation["resource_mapping"].items()))
            require(type(operation.get("resource_mappings")) is list)
            for value in operation["resource_mappings"]:
                mapping(value)
            require(type(operation.get("definition_events")) is list and type(operation.get("original_arguments")) is list)
            for event in operation["definition_events"]:
                require(type(event) is dict and event.get("kind") == "definition_change" and type(event.get("formal_resource")) is str)
                require(event.get("resource") is None or type(event["resource"]) is str)
                require(strings(event.get("guard")) and type(event.get("position")) is str)
            for argument in operation["original_arguments"]:
                require(type(argument) is dict and type(argument.get("position")) is int and argument["position"] >= 0
                        and type(argument.get("formal")) is str and type(argument.get("actual")) is str)
                require(argument.get("keyword") is None or type(argument["keyword"]) is str)
        elif operation["kind"] == "native_contract":
            require(all(type(operation.get(key)) is str for key in ("procedure", "identity", "contract_sha256")))
            require(type(operation.get("effects")) is list)
            for effect in operation["effects"]:
                require(type(effect) is dict and effect.get("kind") in {"read", "write", "overwrite"}
                        and type(effect.get("resource")) is str and type(effect.get("rank")) is int and effect["rank"] >= 0)
                require(effect.get("section") == "whole")
        else:
            require(type(operation.get("source")) is str)
    if routine.source_kind == "external":
        require(item["cloneable"] is False and type(item.get("call_interface")) is dict
                and type(item["call_interface"].get("available")) is bool)


@dataclass
class Binding:
    name: str
    root: str
    dtype: str
    kind: int | None
    rank: int
    intent: str | None = None
    attributes: frozenset[str] = frozenset()
    lower_bounds: tuple[str, ...] = ()
    lower_bound_nodes: tuple[object | None, ...] = field(default=(), repr=False)
    shape_nodes: tuple[object, ...] = field(default=(), repr=False)
    declaring_scope: object = field(default=None, repr=False, compare=False)

    def signature(self):
        return self.dtype, self.kind, self.rank

    def public(self):
        return {"name": self.name, "resource": self.root, "type": self.dtype,
                "kind": self.kind, "rank": self.rank, "intent": self.intent,
                "attributes": sorted(self.attributes), "lower_bounds": self.lower_bounds,
                "shape": [{"kind": _kind(axis), "source": str(axis),
                           "bounds": ([str(axis.items[1]) if axis.items[1] is not None else None, "*"]
                                      if _kind(axis) == "Assumed_Size_Spec" else
                                      [str(item) if item is not None else None for item in axis.items])}
                          for axis in self.shape_nodes]}


@dataclass
class Scope:
    module: str
    path: Path
    node: object
    parent: Scope | None = None
    kinds: _KindScope | None = None
    bindings: dict[str, Binding] = field(default_factory=dict)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)
    wildcards: list[str] = field(default_factory=list)
    wildcard_exclusions: dict[str, set[str]] = field(default_factory=dict)
    ambiguous_imports: set[str] = field(default_factory=set)
    generics: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    unresolved_imports: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    default_public: bool = True
    access: dict[str, bool] = field(default_factory=dict)
    externals: set[str] = field(default_factory=set)
    procedure_arguments: set[str] = field(default_factory=set)
    # Lexical procedure bindings are distinct from module exports. In
    # particular, equal helper names in two CONTAINS owners must not collide.
    procedures: dict[str, str] = field(default_factory=dict)
    intrinsic_procedures: dict[str, str] = field(default_factory=dict)
    intrinsic_wildcards: set[str] = field(default_factory=set)
    intrinsic_exclusions: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class Routine:
    scope: Scope
    arguments: tuple[str, ...]
    execution: object
    qualified: str
    issues: list[str]
    source_kind: str = "module"


@dataclass
class _Closure:
    summaries: dict = field(default_factory=dict)
    operations: int = 0


class SourceEffects:
    def __init__(self, paths, *, contracts=None, depth=8, procedures=64, operations=256, analysis_sources=None,
                 summary_cache=None):
        if min(depth, procedures, operations) < 1:
            raise CompilationError("source effect budgets must be positive")
        self.depth_limit, self.procedure_limit, self.operation_limit = depth, procedures, operations
        self.modules, self.routines, self.summaries = {}, {}, {}
        self.functions = set()
        self.numerical_helpers = {}
        self._closures = {}
        self._native_sections = {}
        self._structures, self._segments, self._descriptor_proofs = {}, {}, {}
        self._joined_completions = {}
        self._native_environments, self._native_environment_states = {}, {}
        self._numerical_completions = {}
        self._worksharing_completions = {}
        self._worksharing_native_completions = {}
        self._reduction_proofs, self._reduction_candidates = {}, {}
        self._omp_reduction_proofs = {}
        self._source_scopes, self._associate_scopes = {}, {}
        self._analysis_started = False
        self._stable_module_allocatables = frozenset()
        self._allocation_authorizations = {}
        self.inputs = SourceInputs(paths,analysis_sources)
        self.sources, self.operation_count = dict(self.inputs.sources), 0
        self.kind_expressions = []
        self.contracts = deepcopy(contracts or {})
        if not isinstance(self.contracts, dict):
            raise CompilationError("effect contracts must be an object")
        try:
            _canonical(self.contracts)
        except (TypeError, ValueError) as error:
            raise CompilationError("effect contracts require finite JSON facts") from error
        self._summary_cache = _DEFAULT_SUMMARY_CACHE if summary_cache is None else SummaryCache(summary_cache)
        self._cache_started = self._summary_cache.stats
        self._cache_rejections = 0
        self._closure_authority = None
        external_nodes = []
        for filename in self.inputs.paths:
            tree = self.inputs.parse(filename)
            for node in _children(tree):
                if _kind(node) == "Subroutine_Subprogram":
                    external_nodes.append((filename, node))
                elif _kind(node) == "Function_Subprogram":
                    name = "$external::" + str(_part(node, "Function_Stmt").items[1]).lower()
                    if name in self.functions:
                        raise CompilationError("duplicate external source procedure " + name)
                    self.functions.add(name)
            for node in walk(tree):
                if _kind(node) != "Module":
                    continue
                module = str(_part(node, "Module_Stmt").items[1]).lower()
                if module in self.modules:
                    raise CompilationError(f"duplicate source module {module}")
                self.modules[module] = Scope(module, filename, node)
        for scope in self.modules.values():
            scope.issues = self._specification(scope)
            if scope.module == "argument":
                scope.issues.append("source module name argument collides with canonical dummy resources")
        for scope in self.modules.values():
            for node in _children(_part(scope.node, "Module_Subprogram_Part")):
                if _kind(node) == "Function_Subprogram":
                    self._register_function(scope, node)
                    continue
                if _kind(node) != "Subroutine_Subprogram":
                    continue
                self._register_routine(scope, node)
        for filename, node in external_nodes:
            self._register_routine(Scope("$external", filename, node), node, source_kind="external")
        for scope in self.modules.values():
            self._import_kinds(scope)
        for routine in self.routines.values():
            self._import_kinds(routine.scope)
        for routine in self.numerical_helpers.values():
            self._import_kinds(routine.scope)
        for binding, scope, selector in self.kind_expressions:
            # An unresolved kind cannot select a generic overload.
            with suppress(CompilationError):
                binding.kind = scope.kinds.integer(selector, SourceLocation(str(scope.path)))
        self._source_roles = {name: (routine.execution, self._routine_signature(routine), str(routine.execution))
                              for name, routine in self.routines.items()}
        self._numerical_roles = {name: (routine.scope.node, self._routine_signature(routine), str(routine.scope.node))
                                 for name, routine in self.numerical_helpers.items()}
        from compiler.frontend.component_bindings import register_associates
        for routine in self.numerical_helpers.values():
            register_associates(self, routine)
        self._source_module_roles = {name: (scope.node, str(scope.node)) for name, scope in self.modules.items()}
        self._source_provenance = {}
        for routine in self.numerical_helpers.values():
            for node in walk(routine.scope.node):
                item = getattr(node, "item", None)
                if item is not None:
                    self._source_provenance[id(node)] = (node, item, item.span,
                                                         getattr(item, "fort_original_span", None))

    def _register_routine(self, parent, node, *, source_kind="module"):
        _attach_execution_directives(node)
        stmt = _part(node, "Subroutine_Stmt")
        owner = getattr(parent, "qualified", parent.module)
        qualified = owner + "::" + str(stmt.items[1]).lower()
        if qualified in self.routines or qualified in self.functions:
            raise CompilationError("duplicate source procedure " + qualified)
        child = Scope(parent.module, parent.path, node, parent if source_kind != "external" else None)
        child.qualified = qualified
        parent.procedures[str(stmt.items[1]).lower()] = qualified
        arguments = tuple(str(argument).lower() for argument in _children(stmt.items[2]))
        issues = parent.issues + self._specification(child, arguments, qualified)
        routine = Routine(child, arguments, _part(node, "Execution_Part"), qualified, issues, source_kind)
        self.routines[qualified] = routine
        self.numerical_helpers[qualified] = routine
        self._register_internal(child, node)

    def _register_function(self, parent, node):
        _attach_execution_directives(node)
        stmt = _part(node, "Function_Stmt")
        qualified = getattr(parent, "qualified", parent.module) + "::" + str(stmt.items[1]).lower()
        if qualified in self.routines or qualified in self.functions:
            raise CompilationError("duplicate source procedure " + qualified)
        self.functions.add(qualified)
        parent.procedures[str(stmt.items[1]).lower()] = qualified
        child = Scope(parent.module, parent.path, node, parent)
        child.qualified = qualified
        arguments = tuple(str(argument).lower() for argument in _children(stmt.items[2]))
        issues = parent.issues + self._specification(child, arguments, qualified)
        prefix_type = next((item for item in _children(stmt.items[0]) if _kind(item) == "Intrinsic_Type_Spec"), None)
        result = str(stmt.items[3].items[0] if stmt.items[3] is not None else stmt.items[1]).lower()
        if prefix_type is not None and result not in child.bindings:
            base, selector = prefix_type.items
            base = str(base).lower()
            width = 8 if base == "double precision" else 4
            if base == "double precision":
                base = "real"
            if selector is not None:
                width = None
                with suppress(CompilationError):
                    width = child.kinds.integer(selector.items[1], SourceLocation(str(child.path)))
            binding = Binding(result, qualified + "::" + result, base, width, 0)
            child.bindings[result] = binding
            if selector is not None:
                self.kind_expressions.append((binding, child, selector.items[1]))
        self.numerical_helpers[qualified] = Routine(child, arguments, _part(node, "Execution_Part"), qualified,
                                                    issues, "internal_function" if parent.parent else "function")
        self._register_internal(child, node)

    def _register_internal(self, parent, node):
        for child in _children(_part(node, "Internal_Subprogram_Part")):
            if _kind(child) == "Subroutine_Subprogram":
                self._register_routine(parent, child, source_kind="internal")
            elif _kind(child) == "Function_Subprogram":
                self._register_function(parent, child)

    @staticmethod
    def external_interface(routine):
        """Admit only source-proven ordinary implicit-interface Fortran ABIs.

        Available implementation source is not itself an explicit caller
        interface. Descriptor dummies, keyword calls and changed calling
        conventions must wait for a proof of that original interface.
        """
        statement = _part(routine.scope.node, "Subroutine_Stmt")
        prefix, _name, _arguments, suffix = statement.items
        reason = None
        if suffix is not None or any(str(item).lower() == "elemental" for item in _children(prefix)):
            reason = "external calling convention requires a proven explicit interface"
        forbidden = {"optional", "allocatable", "pointer", "value", "contiguous", "asynchronous", "volatile"}
        for name in routine.arguments:
            binding = routine.scope.bindings.get(name)
            if binding is None or binding.dtype not in {"real", "integer", "logical"} or binding.kind is None:
                reason = reason or "external formal type requires a proven explicit interface"
            elif (binding.attributes & forbidden or any(
                    _kind(axis) != "Explicit_Shape_Spec" and not (
                        _kind(axis) == "Assumed_Size_Spec" and index == binding.rank-1)
                    for index, axis in enumerate(binding.shape_nodes))):
                reason = reason or "external descriptor or shape requires a proven explicit interface"
        return {"available": reason is None, "kind": "source-proven implicit interface" if reason is None else "explicit interface required",
                "reason": reason, "caller_interface_proof": False, "keyword_arguments": False}

    @staticmethod
    def _routine_signature(routine):
        return _canonical({"arguments": routine.arguments, "issues": routine.issues, "source_kind": routine.source_kind,
                           "bindings": {name: binding.public() for name, binding in routine.scope.bindings.items()},
                           "lexical_procedures": dict(routine.scope.procedures),
                           "intrinsic_procedures": dict(routine.scope.intrinsic_procedures),
                           "intrinsic_wildcards": sorted(routine.scope.intrinsic_wildcards),
                           "intrinsic_exclusions": {module: sorted(names) for module, names in
                                                    routine.scope.intrinsic_exclusions.items()},
                           "imports": routine.scope.imports,
                           "wildcard_exclusions": {module: sorted(names) for module, names in
                                                   routine.scope.wildcard_exclusions.items()},
                           "ambiguous_imports": sorted(routine.scope.ambiguous_imports),
                           "logical_bound_nodes": {name: [str(node) if node is not None else None for node in binding.lower_bound_nodes]
                                                   for name, binding in routine.scope.bindings.items()}})

    def _summary_authority(self):
        from compiler.frontend.structured_effects import STRUCTURED_EFFECT_VERSION
        projections = []
        for name, routine in self.routines.items():
            original = self._source_roles.get(name)
            signature = self._routine_signature(routine)
            if (original is None or routine.execution is not original[0] or signature != original[1]
                    or str(routine.execution) != original[2]):
                projections.append({"procedure": name, "signature": signature,
                                    "execution": str(routine.execution)})
        return {"summary_version": SOURCE_SUMMARY_VERSION, "sources": self.inputs.identity(),
                "structured_version": STRUCTURED_EFFECT_VERSION,
                "native_metadata_version": 1 if getattr(self, '_native_metadata', False) else None,
                "contracts": self.contracts, "capture_authorizations": self._allocation_authorizations,
                "stable_module_allocatables": sorted(self.stable_module_allocatables),
                "budgets": {"depth": self.depth_limit, "procedures": self.procedure_limit,
                            "operations": self.operation_limit},
                "role": "private_projection" if projections else "source",
                "projections": projections}

    def _stamp_summary(self, summary):
        authority = self._summary_authority()
        summary["summary_version"] = SOURCE_SUMMARY_VERSION
        summary["summary_role"] = authority["role"]
        summary["analysis_identity"] = sha256(_canonical(authority).encode()).hexdigest()
        body = {key: value for key, value in summary.items() if key != "summary_identity"}
        summary["summary_identity"] = sha256(_canonical(body).encode()).hexdigest()
        return summary

    def _cached_closure(self, requested):
        authority = self._summary_authority()
        # A structured fragment replaces execution/INTENT under its original
        # qualified name. Its private AST proof must not borrow a source closure.
        if authority["role"] != "source":
            return None
        payload = self._summary_cache.lookup(authority, requested)
        if payload is None:
            return None
        try:
            summaries = payload["summaries"]
            if (requested not in summaries or len(summaries) > self.procedure_limit
                    or payload["operations"] > self.operation_limit
                    or payload["operations"] != sum(len(item["operations"]) for item in summaries.values())):
                raise ValueError("cached closure exceeds proof budget")
            identity = sha256(_canonical(authority).encode()).hexdigest()
            for name, item in summaries.items():
                if name not in self.routines:
                    raise ValueError("cached procedure is unavailable")
                _admit_cached_summary(item, self.routines[name], self.operation_limit)
                if (name not in self.routines or item.get("analysis_identity") != identity
                        or item.get("summary_version") != SOURCE_SUMMARY_VERSION
                        or item.get("summary_role") != "source"
                        or item["closure_depth"] > self.depth_limit):
                    raise ValueError("cached closure identity or depth differs")
                body = {key: value for key, value in item.items() if key != "summary_identity"}
                if sha256(_canonical(body).encode()).hexdigest() != item.get("summary_identity"):
                    raise ValueError("cached summary digest differs")
            depths = {}

            def depth(name, active=()):
                if name in active or name not in summaries:
                    raise ValueError("cached call graph is recursive or incomplete")
                if name not in depths:
                    children = [op for op in summaries[name]["operations"] if op["kind"] == "call"]
                    for op in children:
                        child = summaries.get(op["procedure"])
                        if child is None or op.get("summary_identity") != child["summary_identity"]:
                            raise ValueError("cached call edge identity differs")
                    depths[name] = max((1 + depth(op["procedure"], (*active, name)) for op in children), default=1)
                    if depths[name] != summaries[name]["closure_depth"]:
                        raise ValueError("cached call depth differs")
                return depths[name]

            if depth(requested) > self.depth_limit or len(depths) != len(summaries):
                raise ValueError("cached closure reachability differs")
            order = payload.get("order", [requested, *sorted(set(summaries) - {requested})])
            return _Closure({name: summaries[name] for name in order}, payload["operations"])
        except (KeyError, TypeError, ValueError, RecursionError):
            self._cache_rejections += 1
            return None

    def _store_closures(self, closure):
        authority = self._summary_authority()
        if authority["role"] != "source":
            return
        # Store reusable reachable leaves as well as their callers. Every import
        # still re-admits the original distinct-procedure and operation budget.
        for requested in reversed(closure.summaries):
            selected = {}

            def visit(name, selected=selected):
                if name in selected:
                    return True
                summary = closure.summaries.get(name)
                if (summary is None or not summary["complete"]
                        or any(operation["kind"] == "native_environment" for operation in summary["operations"])):
                    # Registered native-state/environment authority is rebuilt
                    # from original calls; serialized summaries never issue it.
                    return False
                selected[name] = summary
                return all(visit(op["procedure"]) for op in summary["operations"] if op["kind"] == "call")

            if visit(requested):
                self._summary_cache.store(authority, requested,
                    {"summaries": selected, "order": list(selected),
                     "operations": sum(len(item["operations"]) for item in selected.values())})

    @property
    def stable_module_allocatables(self):
        """Canonical roots with caller-provided, source-bound lifetime proofs."""
        return self._stable_module_allocatables

    def authorize_stable_module_allocatables(self, roots):
        """Accept bounded lifetime proofs before any effect or section queries.

        The source-scope caller must first validate its source hashes and stable,
        nonescaping allocation facts. This authorizes borrowing these module
        arrays only; it does not prove bounds, definitions, aliasing, completion,
        participation, or operations which can change allocation or association.
        Standalone effect analysis remains conservative by default.
        """
        if self._analysis_started:
            raise CompilationError("module allocation lifetime authorization must precede source analysis")
        if not isinstance(roots, (set, frozenset)) or len(roots) > self.operation_limit:
            raise CompilationError("module allocation lifetime authorization requires a bounded root set")
        if any(not isinstance(root, str) for root in roots):
            raise CompilationError("module allocation lifetime authorization requires canonical source roots")
        numeric = {("real", 4), ("real", 8), ("integer", 4), ("logical", 1)}
        forbidden = {"pointer", "optional", "volatile", "asynchronous", "value", "parameter"}
        authorizations = {}
        for root in sorted(roots):
            parts = root.split("::")
            module = self.modules.get(parts[0]) if len(parts) == 2 and parts[0] != "argument" else None
            binding = module.bindings.get(parts[1]) if module else None
            if (binding is None or binding.root != root or binding.rank < 1
                    or "allocatable" not in binding.attributes or binding.attributes & forbidden
                    or (binding.dtype, binding.kind) not in numeric):
                raise CompilationError("stable allocation proof requires a supported canonical module array: " + root)
            authorizations[root] = {"resource": root, "source": str(module.path),
                                    "source_sha256": self.sources[str(module.path)]}
        self.inputs.verify()
        # Commit only after every root and the original input identities pass.
        self._stable_module_allocatables = frozenset(roots)
        self._allocation_authorizations = authorizations

    def _import_kinds(self, scope, active=frozenset()):
        if id(scope) in active:
            return
        active = active | {id(scope)}
        imported = []

        def exports(module, visited=frozenset()):
            target = self.modules.get(module)
            if target is None or module in visited:
                return set()
            names = set(target.bindings) | set(target.imports)
            for parent in target.wildcards:
                names.update(exports(parent, visited | {module}))
            return {name for name in names if self._exported(target, name)}

        for local, (module, remote) in scope.imports.items():
            imported.append((local, module, remote))
        for module in scope.wildcards:
            target = self.modules.get(module)
            if target:
                imported += [(name, module, name) for name in sorted(exports(module))]
        for local, module, remote in imported:
            target = self.modules.get(module)
            if target is None or not self._exported(target,remote):
                continue
            # Resolve the complete local import set. Choosing one declaration
            # here would silently price/lower an ambiguous imported kind.
            binding = self._binding(scope, local)
            if binding is None or binding.dtype != "integer" or binding.rank or "parameter" not in binding.attributes:
                continue
            owner = self.modules.get(binding.root.split("::",1)[0])
            if owner is None:
                continue
            self._import_kinds(owner, active)
            try:
                value = owner.kinds.integer(binding.name, SourceLocation(str(owner.path)))
            except CompilationError:
                continue
            if local not in scope.bindings:
                scope.kinds.values[local] = value

    def _specification(self, scope, arguments=(), qualified=None):
        spec = _part(scope.node, "Specification_Part")
        scope.kinds = _KindScope(spec, scope.parent.kinds if scope.parent else None)
        issues = []
        for node in _children(spec):
            name = _kind(node)
            if name == "Use_Stmt":
                nature, _, module, only, symbols = node.items
                module = str(module).lower()
                if (module in {"ieee_arithmetic", "ieee_exceptions", "ieee_features"}
                        and str(nature).lower() != "non_intrinsic"
                        and (str(nature).lower() == "intrinsic" or module not in self.modules)):
                    # Standard intrinsic modules do not export mutable user
                    # storage which could shadow a host capture. Their calls
                    # still require explicit numerical/effect support below.
                    unrestricted = symbols is None or str(only).upper().replace(" ", "") != ",ONLY:"
                    renamed = set()
                    if symbols is not None:
                        for item in symbols.items:
                            local, remote = ((str(item.items[1]).lower(), str(item.items[2]).lower())
                                             if _kind(item) == "Rename" else (str(item).lower(), str(item).lower()))
                            if _kind(item) == "Rename":
                                renamed.add(remote)
                            identity = "$intrinsic::" + module + "::" + remote
                            previous = scope.intrinsic_procedures.get(local)
                            if previous is not None and previous != identity:
                                from compiler.frontend.native_environment import canonical_intrinsic_export
                                if (canonical_intrinsic_export(previous) is None
                                        or canonical_intrinsic_export(previous) != canonical_intrinsic_export(identity)):
                                    scope.ambiguous_imports.add(local)
                            scope.intrinsic_procedures[local] = identity
                    if unrestricted:
                        scope.intrinsic_wildcards.add(module)
                        if module in scope.intrinsic_exclusions:
                            scope.intrinsic_exclusions[module].intersection_update(renamed)
                        else:
                            scope.intrinsic_exclusions[module] = renamed
                    continue
                if module in _KindScope.intrinsic_kinds and str(nature).lower() != "non_intrinsic":
                    continue
                unrestricted = symbols is None or str(only).upper().replace(" ", "") != ",ONLY:"
                renamed = set()
                if symbols is not None:
                    for item in symbols.items:
                        local, remote = ((str(item.items[1]), str(item.items[2]))
                                         if _kind(item) == "Rename" else (str(item), str(item)))
                        local, remote = local.lower(), remote.lower()
                        if _kind(item) == "Rename":
                            renamed.add(remote)
                        previous = scope.imports.get(local)
                        if previous is not None and previous != (module, remote):
                            # Do not arbitrarily choose between explicit local
                            # names. A later canonical-export merge can admit
                            # aliases that are proven to name the same entity.
                            scope.ambiguous_imports.add(local)
                        scope.imports[local] = module, remote
                if unrestricted:
                    if module in scope.wildcard_exclusions:
                        scope.wildcard_exclusions[module].intersection_update(renamed)
                    else:
                        scope.wildcards.append(module)
                        scope.wildcard_exclusions[module] = renamed
                if module not in self.modules:
                    scope.unresolved_imports.append(module)
            elif name == "Interface_Block":
                start = _part(node, "Interface_Stmt")
                generic = str(start.items[0]).lower() if start.items[0] is not None else None
                members = []
                for item in _children(node):
                    if _kind(item) == "Procedure_Stmt":
                        members += [(scope.module, str(p).lower()) for p in item.items[0].items]
                if generic and members:
                    scope.generics[generic] = members
                else:
                    issues.append("non-module generic interface is unavailable")
            elif name == "Procedure_Declaration_Stmt":
                interface, attributes, entities = node.items
                if (_kind(interface) == "Name" and attributes is None
                        and all(_kind(item) == "Name" for item in _children(entities))):
                    # Merely declaring an ordinary dummy procedure executes
                    # nothing. An invocation still needs its own resolved
                    # source target; never infer its effects from this type.
                    names = {str(item).lower() for item in _children(entities)}
                    scope.procedure_arguments.update(names.intersection(arguments))
                    scope.externals.update(names)
                else:
                    issues.append("procedure association or initialization requires source proof")
            elif name == "External_Stmt":
                names = {str(item).lower() for item in _children(node.items[-1])}
                scope.externals.update(names)
                scope.procedure_arguments.update(names.intersection(arguments))
            elif name == "Access_Stmt":
                mode, names = node.items
                public = str(mode).lower() == "public"
                if names is None:
                    scope.default_public = public
                else:
                    scope.access.update((str(item).lower(),public) for item in names.items)
            elif name == "Type_Declaration_Stmt":
                dtype, attrs, entities = node.items
                if _kind(dtype) == "Intrinsic_Type_Spec":
                    base, selector = dtype.items
                    base = str(base).lower()
                    width = 8 if base == "double precision" else 4
                    if selector is not None:
                        try:
                            width = scope.kinds.integer(selector.items[1], SourceLocation(str(scope.path)))
                        except CompilationError:
                            width = None
                    if base == "double precision":
                        base = "real"
                else:
                    base, width = str(dtype).lower(), None
                attrs = tuple(_children(attrs))
                flags = frozenset(str(a).split("(")[0].lower() for a in attrs)
                dimension = next((a.items[1] for a in attrs if _kind(a) == "Dimension_Attr_Spec"), None)
                intent = next((str(a.items[1]).lower() for a in attrs if _kind(a) == "Intent_Attr_Spec"), None)
                for entity in entities.items:
                    variable, shape, _, initializer = entity.items
                    variable = str(variable).lower()
                    shape = shape if shape is not None else dimension
                    dimensions = (tuple(_children(shape.items[0])) + (shape,) if _kind(shape) == "Assumed_Size_Spec"
                                  else tuple(_children(shape)))
                    lower_nodes = tuple(d.items[1] if _kind(d) == "Assumed_Size_Spec" else d.items[0]
                                        for d in dimensions)
                    lowers = tuple(str(lower) if lower is not None else "1" for lower in lower_nodes)
                    persistent = "save" in flags or (initializer is not None and "parameter" not in flags)
                    attributes = flags | ({"save"} if persistent else set())
                    root = (f"argument::{variable}" if variable in arguments else
                            f"{qualified}::{variable}" if qualified else f"{scope.module}::{variable}")
                    binding = Binding(variable, root, base, width, len(dimensions), intent,
                                      frozenset(attributes), lowers, lower_nodes, dimensions, declaring_scope=scope)
                    scope.bindings[variable] = binding
                    if _kind(dtype) == "Intrinsic_Type_Spec" and selector is not None:
                        self.kind_expressions.append((binding, scope, selector.items[1]))
            elif name not in {"Implicit_Part", "Comment", "Public_Stmt", "Private_Stmt", "Access_Stmt", "Derived_Type_Def"}:
                issues.append(f"specification effect unavailable: {name}")
        from compiler.frontend.component_bindings import register_types
        register_types(self, scope, spec)
        return issues

    def _exported(self, scope, name):
        name = str(name).lower()
        if name in scope.access:
            return scope.access[name]
        binding = scope.bindings.get(name)
        if binding and "private" in binding.attributes:
            return False
        if binding and "public" in binding.attributes:
            return True
        return scope.default_public

    def use_sources(self, scope, name):
        """Visible USE candidates, including exclusions on unrestricted renames.

        Repeated unrestricted imports intersect exclusions: another USE can
        make the original name visible again. Explicit ONLY imports remain
        available independently of those exclusions.
        """
        if name in scope.ambiguous_imports:
            return None
        choices = ([scope.imports[name]] if name in scope.imports else [])
        choices += [(module, name) for module in scope.wildcards
                    if name not in scope.wildcard_exclusions.get(module, ())]
        return tuple(dict.fromkeys(choices))

    def _binding(self, scope, name, visited=frozenset()):
        if _kind(name) == "Data_Ref":
            from compiler.frontend.component_bindings import resolve_binding
            return resolve_binding(self, scope, name)
        name = str(name).lower()
        key = (scope.module, id(scope), name)
        if key in visited:
            return None
        if name in scope.bindings:
            return scope.bindings[name]
        if name in scope.intrinsic_procedures or (name.startswith("ieee_") and scope.intrinsic_wildcards):
            return None
        found = []
        sources = self.use_sources(scope, name)
        if sources is None:
            return None
        for module, remote in sources:
            if module in self.modules:
                if not self._exported(self.modules[module], remote):
                    if name in scope.imports and scope.imports[name] == (module, remote):
                        return None
                    continue
                binding = self._binding(self.modules[module], remote, visited | {key})
                if binding:
                    found.append(binding)
            else:
                return None  # A local USE name shadows any parent binding.
        roots = {b.root: b for b in found}
        if roots:
            return next(iter(roots.values())) if len(roots) == 1 else None
        return self._binding(scope.parent, name, visited | {key}) if scope.parent else None

    def _candidates(self, scope, name, visited=frozenset()):
        name = str(name).lower()
        key = (scope.module, id(scope), name)
        if key in visited:
            return []
        if name in scope.bindings or name in scope.procedure_arguments:
            return []
        if name in scope.intrinsic_procedures:
            return [scope.intrinsic_procedures[name]]
        if name.startswith("ieee_") and scope.intrinsic_wildcards:
            return ["$intrinsic::" + sorted(scope.intrinsic_wildcards)[0] + "::" + name]
        if name in scope.procedures:
            return [scope.procedures[name]]
        external = "$external::" + name
        if name in scope.externals:
            return [external] if external in self.routines or external in self.functions else []
        if name in scope.generics:
            return [m + "::" + p for m, p in scope.generics[name]]
        sources = self.use_sources(scope, name)
        if sources is None:
            return []
        own = scope.module + "::" + name
        if scope.parent is None and scope.module in self.modules and (own in self.routines or own in self.functions):
            return [own]
        found = []
        for module, remote in sources:
            if module not in self.modules:
                if module in scope.wildcards:
                    return []
                found.append(module + "::" + remote)
                continue
            if not self._exported(self.modules[module],remote):
                if name in scope.imports and scope.imports[name] == (module, remote):
                    return []
                continue
            found += self._candidates(self.modules[module], remote, visited | {key})
        if found:
            return sorted(set(found))
        if scope.parent is not None:
            return self._candidates(scope.parent, name, visited | {key})
        return [external] if external in self.routines or external in self.functions else []

    def resource_identity_boundary(self, binding):
        """Reserved formal roots cannot also identify defining module storage."""
        module = self.modules.get("argument")
        if module is not None and any(binding is value for value in module.bindings.values()):
            return "source module name argument collides with canonical dummy resources: " + binding.root
        return None

    def _actual_binding(self, scope, node):
        if _kind(node) == "Name":
            return self._binding(scope, node)
        if _kind(node) == "Data_Ref":
            from compiler.frontend.component_bindings import component_access
            access = component_access(self, scope, node)
            if access is not None and not access.indices:
                return access.binding
        return None

    def _actual_mapping_boundary(self, scope, node):
        # An element actual is scalar at the callee, but reads/writes storage
        # belonging to its array root at the caller. It needs coherence before
        # evaluating its address/value and an expression mapping when owner
        # captures acquire private names. Whole-variable mappings do neither.
        for part in walk(node):
            if _kind(part) == "Part_Ref":
                binding = self._binding(scope, part.items[0])
                if binding and binding.rank:
                    return "array-element/section actual requires in-place mapping and coherence: " + str(node)
        return None

    def _unknown_exports(self, scope):
        # An unavailable wildcard USE can supply a procedure with the same
        # spelling as an intrinsic. A known local argument says nothing about
        # which implementation of SUM/SIZE/etc. was invoked.
        while scope is not None:
            if any(module not in self.modules for module in scope.wildcards):
                return True
            scope = scope.parent
        return False

    def _signature(self, scope, node, _depth=0):
        """Prove a bounded actual's type without evaluating its expression.

        Scalar arithmetic keeps its original Fortran call position and effects.
        This does not authorize reassociation, early bound evaluation or a
        writable temporary actual; those remain separate call proofs.
        """
        if _depth > self.depth_limit:
            return None
        binding = self._actual_binding(scope, node)
        if binding:
            return binding.signature()
        items = _children(node)
        if _kind(node) == "Parenthesis":
            return self._signature(scope, items[1], _depth + 1)
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            child = self._signature(scope, items[1], _depth + 1)
            return child if child in {("integer", 4, 0), ("integer", 8, 0),
                                      ("real", 4, 0), ("real", 8, 0)} else None
        if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/"}:
            left = self._signature(scope, items[0], _depth + 1)
            right = self._signature(scope, items[2], _depth + 1)
            return left if left == right and left in {
                ("integer", 4, 0), ("integer", 8, 0), ("real", 4, 0), ("real", 8, 0)} else None
        if _kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
            dtype = {"Int_Literal_Constant": "integer", "Real_Literal_Constant": "real",
                     "Logical_Literal_Constant": "logical"}[_kind(node)]
            width = 8 if dtype == "real" and "d" in node.items[0].lower() else 4
            if node.items[1] is not None:
                try:
                    width = scope.kinds.integer(node.items[1], SourceLocation(str(scope.path)))
                except CompilationError:
                    width = None
            return dtype, width, 0
        return None

    def whole_overwrites(self, routine):
        """Prove complete assignments, including unit-stride full-array sweeps.

        This is a must-write proof, separate from the conservative may-write
        envelope. Uncertain bounds, conditional holes, extra loop iterators and
        early exits supply no proof. SIZE-based sweeps require dummy lower bound
        one; LBOUND/UBOUND sweeps also support negative declared lower bounds.
        """
        scope = routine.scope

        def normalized(node):
            return str(node).lower().replace(" ", "")

        def inquiry(node, name, binding, axis):
            if _kind(node) != "Intrinsic_Function_Reference" or str(node.items[0]).lower() != name:
                return False
            arguments = _children(node.items[1])
            if len(arguments) != 2 or _kind(arguments[0]) != "Name":
                return False
            array = self._binding(scope, arguments[0])
            return array is not None and array.root == binding.root and normalized(arguments[1]) == str(axis)

        def sweep(control, binding, axis):
            lower, upper, *steps = control
            if steps and steps[0] is not None and normalized(steps[0]) != "1":
                return False
            declared = binding.lower_bounds[axis-1].replace(" ", "").lower()
            # Deferred-shape allocatables retain allocation-time lower bounds.
            # The declaration's omitted lower origin is not a runtime value.
            fixed_origin = "allocatable" not in binding.attributes
            lo = inquiry(lower, "lbound", binding, axis) or (fixed_origin and normalized(lower) == declared)
            hi = inquiry(upper, "ubound", binding, axis) or (
                fixed_origin and declared == "1" and inquiry(upper, "size", binding, axis))
            return lo and hi

        def full_target(target, loops):
            if _kind(target) == "Name" and not loops:
                binding = self._binding(scope, target)
                return binding if binding and binding.rank else None
            if _kind(target) != "Part_Ref":
                return None
            binding = self._binding(scope, target.items[0])
            indices = tuple(_children(target.items[1]))
            if not binding or not binding.rank or len(indices) != binding.rank:
                return None
            used = set()
            for axis, index in enumerate(indices, 1):
                if _kind(index) == "Subscript_Triplet" and all(v is None for v in index.items):
                    continue
                if _kind(index) != "Name":
                    return None
                iterator = normalized(index)
                if iterator in used or iterator not in loops or not sweep(loops[iterator], binding, axis):
                    return None
                used.add(iterator)
            return binding if used == set(loops) else None

        def prove(nodes, loops=None):
            loops = {} if loops is None else loops
            nodes = [node for node in nodes if _kind(node) != "Comment"]
            # A RETURN/EXIT/call could skip a candidate assignment or mutate its
            # sweep bounds. Restrict this proof to complete assignment-only nests.
            if any(_kind(n) not in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct", "Continue_Stmt"}
                   for n in nodes):
                return set()
            found = set()
            for node in nodes:
                if _kind(node) == "Assignment_Stmt":
                    target = full_target(node.items[0], loops)
                    if target:
                        found.add(target.root)
                elif _kind(node) == "Block_Nonlabel_Do_Construct":
                    children = node.content
                    control = next((c for c in _children(children[0]) if _kind(c) == "Loop_Control"), None)
                    if control is None or control.items[1] is None:
                        continue
                    iterator, bounds = control.items[1]
                    iterator = normalized(iterator)
                    if iterator in loops:
                        continue
                    # Any assignment to an active iterator or its bound storage
                    # defeats a static coverage proof.
                    writes = [n.items[0] for n in walk(node) if _kind(n) == "Assignment_Stmt"]
                    bound_names = {str(n).lower() for b in bounds if b is not None
                                   for n in walk(b) if _kind(n) == "Name"}
                    if any(_kind(w) == "Name" and str(w).lower() in bound_names | {iterator}
                           for w in writes):
                        continue
                    found.update(prove(children[1:-1], {**loops, iterator: bounds}))
            return found

        return sorted(prove(_children(routine.execution)))

    def summarize(self, requested, active=(), *, _closure=None):
        self._analysis_started = True
        # Each requested proof owns its budget. Unrelated extraction offers and
        # rejected branches cannot exhaust it or leave apparently complete,
        # empty summaries behind. Cache a closure only in its original context.
        if _closure is None:
            self.inputs.verify()
            authority = _canonical(self._summary_authority())
            if authority != self._closure_authority:
                self._closures.clear()
                self.summaries.clear()
                self._closure_authority = authority
            closure = self._closures.get(requested)
            if closure is None:
                closure = self._cached_closure(requested)
                if closure is None:
                    closure = _Closure()
                    self.summarize(requested, active, _closure=closure)
                    self._store_closures(closure)
                if len(self._closures) < self.procedure_limit:
                    self._closures[requested] = closure
            self.operation_count = closure.operations
            self.summaries.update(closure.summaries)
            return deepcopy(closure.summaries[requested])
        if requested not in self.routines:
            raise CompilationError(f"source procedure unavailable: {requested}")
        if requested in active or len(active) >= self.depth_limit:
            return {"procedure": requested, "complete": False, "cloneable": False, "operations": [],
                    "reasons": ["recursive call or bounded source closure exhausted"]}
        if requested in _closure.summaries:
            cached = _closure.summaries[requested]
            if len(active) + cached["closure_depth"] <= self.depth_limit:
                return cached
            return {"procedure": requested, "complete": False, "cloneable": False, "operations": [],
                    "reasons": ["bounded source closure depth exhausted"]}
        if len(_closure.summaries) >= self.procedure_limit:
            return {"procedure": requested, "complete": False, "cloneable": False, "operations": [],
                    "reasons": ["bounded source closure procedure budget exhausted"]}
        cached_closure = self._cached_closure(requested)
        if cached_closure is not None:
            new = {name: item for name, item in cached_closure.summaries.items() if name not in _closure.summaries}
            added_operations = sum(len(item["operations"]) for item in new.values())
            if (not set(cached_closure.summaries).intersection(active)
                    and len(active) + cached_closure.summaries[requested]["closure_depth"] <= self.depth_limit
                    and len(_closure.summaries) + len(new) <= self.procedure_limit
                    and _closure.operations + added_operations <= self.operation_limit):
                _closure.summaries.update(new)
                _closure.operations += added_operations
                return _closure.summaries[requested]
        routine = self.routines[requested]
        reasons = list(routine.issues)
        operations = []
        summary = {"procedure": requested, "source_kind": routine.source_kind, "complete": False, "cloneable": False,
                   "arguments": [routine.scope.bindings[a].public() for a in routine.arguments
                                 if a in routine.scope.bindings], "operations": operations, "reasons": reasons,
                   "definition_diagnostics": [], "closure_depth": 1}
        if routine.source_kind == "external":
            summary["call_interface"] = self.external_interface(routine)
        lifetime_requirements = {}
        summary["capture_lifetime_requirements"] = []
        _closure.summaries[requested] = summary
        undeclared = set(routine.arguments) - routine.scope.bindings.keys()
        if undeclared:
            reasons.append("undeclared dummy arguments: " + ", ".join(sorted(undeclared)))
        persistent = [b.root for b in routine.scope.bindings.values() if "save" in b.attributes]
        summary["persistent_state"] = persistent
        summary["definition_changes"] = [b.root for b in routine.scope.bindings.values()
                                         if b.intent == "out" and b.name in routine.arguments]
        summary["descriptor_requirements"] = [b.public() for b in routine.scope.bindings.values()
                                               if b.name in routine.arguments
                                               and b.attributes & {"optional", "allocatable", "pointer"}]
        descriptor_proof = (self.descriptor_stability(requested) if any(
            "allocatable" in binding.attributes
            for binding in routine.scope.bindings.values()) else {"resources": []})
        stable_descriptors = {item["resource"] for item in descriptor_proof["resources"] if item["stable"]}
        for binding in routine.scope.bindings.values():
            if (binding.name in routine.arguments and "allocatable" in binding.attributes
                    and binding.intent != "in" and binding.root not in stable_descriptors):
                reasons.append("allocation-changing dummy descriptor semantics: " + binding.root)

        def emit(operation):
            _closure.operations += 1
            if _closure.operations > self.operation_limit:
                if "source operation budget exhausted" not in reasons:
                    reasons.append("source operation budget exhausted")
                return False
            operations.append(operation)
            return True

        def effect(binding, action, guard, spelling):
            if binding is None:
                reasons.append(f"unresolved storage: {spelling}")
                return
            namespace = self.resource_identity_boundary(binding)
            if namespace:
                reasons.append(namespace)
                return
            if "parameter" in binding.attributes:
                return
            if action != "descriptor_read" and {"pointer", "allocatable"} & binding.attributes:
                authorized = binding.root in self.stable_module_allocatables
                readonly_descriptor = (binding.name in routine.arguments and binding.intent == "in"
                                       and "allocatable" in binding.attributes and "pointer" not in binding.attributes)
                stable_descriptor = binding.root in stable_descriptors and "pointer" not in binding.attributes
                lifetime_requirements[binding.root] = {
                    "resource": binding.root, "authorized": authorized or stable_descriptor,
                    **({"requirement": "original_readonly_allocatable_descriptor",
                        "execution_requires_runtime_guard": True} if readonly_descriptor else
                       {"requirement": "original_stable_allocatable_descriptor",
                        "execution_requires_runtime_guard": True} if stable_descriptor else {})}
                if not authorized and not readonly_descriptor and not stable_descriptor:
                    reasons.append(f"storage lifetime requires capture proof: {binding.root}")
                elif action == "overwrite":
                    # A whole allocatable LHS can allocate/reallocate even
                    # without an explicit ALLOCATE statement. Element/section
                    # assignments do not perform that association change.
                    reasons.append(f"whole allocatable assignment may change storage: {binding.root}")
            external = (binding.name in routine.arguments or not binding.root.startswith(requested + "::")
                        or "save" in binding.attributes or binding.root in getattr(self, "_captured_local_roots", ()))
            if external:
                emit({"kind": action, "resource": binding.root, "rank": binding.rank,
                      "section": "whole", "guard": guard, "source_access": spelling})

        def expression(node, guard, metadata=False):
            if node is None or isinstance(node, (str, int)):
                return
            name = _kind(node)
            scope = self.source_scope_for(node, routine.scope)
            if name == "Name":
                binding = self._binding(scope, node)
                if binding is None and metadata:
                    from compiler.frontend.native_environment import intrinsic_export
                    if intrinsic_export(self, scope, node) == "$intrinsic::ieee_exceptions::ieee_all":
                        # The original intrinsic constant supplies its shape;
                        # neither its value nor an assumed extent is evaluated.
                        return
                effect(binding, "descriptor_read" if metadata else "read", guard, str(node))
            elif name == "Part_Ref":
                base, indices = node.items
                binding = self._binding(scope, base)
                if binding is None or not binding.rank:
                    reasons.append(f"unknown function or indexed storage: {node}")
                else:
                    effect(binding, "read", guard, str(node))
                for index in _children(indices):
                    expression(index, guard)
            elif name == "Intrinsic_Function_Reference":
                function, args = node.items
                intrinsic = str(function).lower()
                # A declared/imported name may shadow the intrinsic.
                shadowed = self._binding(scope, intrinsic) or self._candidates(scope, intrinsic)
                if shadowed or self._unknown_exports(scope) or intrinsic not in set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
                    "allocated", "present", "sum", "product", "any", "all", "count", "minval", "maxval"}:
                    reasons.append(f"unresolved function effects: {function}")
                for index, argument in enumerate(_children(args)):
                    if _kind(argument) == "Actual_Arg_Spec":
                        argument = argument.items[1]
                    expression(argument, guard, metadata=index == 0 and intrinsic in ARRAY_INQUIRIES | MODEL_INQUIRIES | {"allocated", "present"})
            elif name.endswith("Literal_Constant"):
                return
            elif name == "Data_Ref":
                from compiler.frontend.component_bindings import component_access
                try:
                    access = component_access(self, scope, node)
                    effect(access.binding, "descriptor_read" if metadata else "read", guard, str(node))
                    for index in access.indices:
                        expression(index, guard)
                except CompilationError as error:
                    reasons.append(str(error))
            elif name in {"Function_Reference", "Structure_Constructor"}:
                reasons.append(f"unresolved component/function effects: {node}")
            else:
                for child in _children(node):
                    expression(child, guard, metadata)

        def call(node, guard):
            target, actuals = node.items
            scope = self.source_scope_for(node, routine.scope)
            if _kind(target) != "Name":
                reasons.append(f"indirect call effects unavailable: {target}")
                return
            actuals = tuple(_children(actuals))
            candidates = self._candidates(scope, target)
            from compiler.frontend.native_environment import intrinsic_export
            intrinsic = intrinsic_export(self, scope, target)
            if intrinsic in {"$intrinsic::ieee_exceptions::ieee_get_halting_mode",
                             "$intrinsic::ieee_exceptions::ieee_set_halting_mode"}:
                try:
                    proof = self.native_environment(requested, node)
                except CompilationError as error:
                    reasons.append(str(error))
                    emit({"kind": "boundary", "call": str(node), "guard": guard, "reason": reasons[-1]})
                    return
                emit({"kind": "native_environment", "intrinsic": intrinsic, "proof_identity": proof.identity,
                      "proof": proof.public(), "effects": proof.public()["effects"], "guard": guard,
                      "source_access": str(node)})
                return
            if any(candidate in self.routines for candidate in candidates):
                try:
                    resolved = resolve_source_call(self, scope, node)
                except CompilationError as error:
                    reasons.append(str(error))
                    emit({"kind": "boundary", "call": str(node), "guard": guard, "reason": reasons[-1]})
                    return
                chosen = resolved.procedure
                callee = self.routines[chosen]
                # Keep descriptor dependencies and section bounds at the call.
                # Passing an array does not itself read its payload. A section's
                # bounds must not be mistaken for a whole-array upload.
                for item in resolved.mappings:
                    if item.section is not None:
                        for dependency in item.section.dependencies:
                            effect(dependency.binding,
                                   "descriptor_read" if dependency.kind == "descriptor_read" else "read",
                                   guard, str(item.actual))
                    elif item.actual is not None and item.binding is None:
                        expression(item.actual, guard)
                    if item.binding is not None and (item.presence != "supplied"
                                                     or "allocatable" in item.formal_binding.attributes):
                        effect(item.binding, "descriptor_read", guard, str(item.actual))
                    if (item.formal_binding.attributes & {"allocatable", "pointer"}
                            and (item.formal_binding.intent != "in"
                                 or "pointer" in item.formal_binding.attributes)
                            and not ("pointer" not in item.formal_binding.attributes
                                     and any(proof["resource"] == "argument::" + item.formal and proof["stable"]
                                             for proof in self.descriptor_stability(chosen)["resources"]))):
                        # INTENT(OUT) can deallocate on entry even for unused
                        # dummies. Borrowed read-only descriptors cannot prove
                        # these allocation or association effects stable.
                        reasons.append(f"allocatable callee formals require original descriptor and allocation semantics: {chosen}")
                child = self.summarize(chosen, active + (requested,), _closure=_closure)
                summary["closure_depth"] = max(summary["closure_depth"], 1 + child.get("closure_depth", 1))
                if not child["complete"]:
                    reasons.append(f"callee effects incomplete: {chosen}")
                summary["definition_diagnostics"].extend(child.get("definition_diagnostics",[]))
                public = resolved.public()
                by_formal = {item["formal_resource"]: item for item in public["resource_mappings"]}
                definition_events = []
                for formal in child.get("definition_changes", ()):
                    item = by_formal.get(formal)
                    if item is not None:
                        definition_events.append({"kind": "definition_change", "formal_resource": formal,
                            "resource": item["resource"], "storage": item["storage"],
                            "section": item["section"], "presence": item["presence"],
                            "guard": guard, "position": "callee entry after original actual evaluation"})
                emit({"kind": "call", "procedure": chosen, "source_kind": callee.source_kind,
                      "actual_arguments": public["actual_arguments"],
                      "original_arguments": public["original_arguments"],
                      "resource_mapping": public["resource_mapping"],
                      "resource_mappings": public["resource_mappings"],
                      "definition_events": definition_events,
                      "summary_identity": child.get("summary_identity"),
                      "guard": guard, "complete": child["complete"]})
            else:
                # A version-bound opaque contract has no source signature.
                # Preserve its existing positional whole-storage interface;
                # keywords, sections and descriptor forwarding need source.
                matches = [candidate for candidate in candidates if candidate in self.contracts]
                actual_boundary = next((reason for actual in actuals
                                        if (reason := self._actual_mapping_boundary(scope, actual))), None)
                if (len(matches) != 1 or len(candidates) != 1
                        or any(_kind(actual) == "Actual_Arg_Spec" for actual in actuals) or actual_boundary):
                    reasons.append(actual_boundary or f"call effects unresolved or ambiguous: {target}")
                    emit({"kind": "boundary", "call": str(node), "guard": guard, "reason": reasons[-1]})
                    return
                chosen = matches[0]
                for actual in actuals:
                    if self._actual_binding(scope, actual) is None:
                        expression(actual, guard)
                contract = self.contracts[chosen]
                if (not isinstance(contract, dict) or contract.get("lifetime") != "stable"
                        or contract.get("escapes") is not False or contract.get("ordering") != "serial"
                        or contract.get("complete") is not True or contract.get("descriptor_changes") is not False
                        or not isinstance(contract.get("identity"), str) or not contract["identity"]
                        or not isinstance(contract.get("effects"), list)):
                    raise CompilationError(f"invalid native effect contract: {chosen}")
                contracted = []
                for item in contract["effects"]:
                    if (not isinstance(item, dict) or item.get("kind") not in {"read", "write", "overwrite"}
                            or item.get("section") != "whole"):
                        raise CompilationError(f"invalid native contract resource effect: {chosen}")
                    if "argument" in item and "resource" not in item:
                        if type(item["argument"]) is not int or not 0 <= item["argument"] < len(actuals):
                            raise CompilationError(f"invalid native contract argument: {chosen}")
                        binding = self._actual_binding(scope, actuals[item["argument"]])
                    elif "resource" in item and "argument" not in item:
                        # Hidden fields need explicit, source-available identity.
                        found = [b for module in self.modules.values() for b in module.bindings.values()
                                 if b.root == item["resource"]]
                        binding = found[0] if len(found) == 1 else None
                    else:
                        raise CompilationError(f"invalid native contract binding: {chosen}")
                    if binding is None:
                        reasons.append(f"contract argument is not stable whole storage: {chosen}")
                        continue
                    namespace = self.resource_identity_boundary(binding)
                    if namespace:
                        reasons.append(namespace)
                        continue
                    if {"pointer", "allocatable"} & binding.attributes:
                        reasons.append(f"contract resource requires capture/lifetime proof: {binding.root}")
                    contracted.append({**item, "resource": binding.root, "rank": binding.rank})
                emit({"kind": "native_contract", "procedure": chosen, "identity": contract["identity"],
                      "contract_sha256": sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                      "effects": contracted, "guard": guard})

        def statements(nodes, guard=()):
            for node in nodes:
                if _closure.operations > self.operation_limit:
                    if "source operation budget exhausted" not in reasons:
                        reasons.append("source operation budget exhausted")
                    return
                kind = _kind(node)
                scope = self.source_scope_for(node, routine.scope)
                if kind == "Assignment_Stmt":
                    target, _, value = node.items
                    expression(value, guard)
                    if _kind(target) == "Name":
                        binding = self._binding(scope, target)
                        effect(binding, "overwrite", guard, str(target))
                    elif _kind(target) == "Part_Ref":
                        effect(self._binding(scope, target.items[0]), "write", guard, str(target))
                        for index in _children(target.items[1]):
                            expression(index, guard)
                    elif _kind(target) == "Data_Ref":
                        from compiler.frontend.component_bindings import component_access
                        try:
                            access = component_access(self, scope, target)
                            effect(access.binding, "write" if access.indices else "overwrite", guard, str(target))
                            for index in access.indices:
                                expression(index, guard)
                        except CompilationError as error:
                            reasons.append(str(error))
                    else:
                        reasons.append(f"unsupported assignment target effects: {target}")
                elif kind == "Call_Stmt":
                    call(node, guard)
                elif kind == "If_Stmt":
                    condition, action = node.items
                    expression(condition, guard)
                    statements((action,), guard + (str(condition),))
                elif kind == "If_Construct":
                    conditions, current = [], []
                    branch = guard
                    for item in node.content:
                        label = _kind(item)
                        if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                            statements(current, branch)
                            current = []
                            branch = guard + tuple(f".not.({c})" for c in conditions)
                            if label in {"If_Then_Stmt", "Else_If_Stmt"}:
                                condition = item.items[0]
                                expression(condition, branch)
                                branch += (str(condition),)
                                conditions.append(str(condition))
                        else:
                            current.append(item)
                elif kind in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}:
                    children = [child for child in node.content if _kind(child) != "Comment"]
                    header = children[0]
                    # Retain loop execution protection as evidence; it is not a
                    # predicate that an adapter may evaluate outside the loop.
                    control = next((item for item in _children(header) if _kind(item) == "Loop_Control"), None)
                    if control is None or control.items[1] is None:
                        reasons.append("unsupported loop control effects")
                    else:
                        iterator, bounds = control.items[1]
                        for bound in bounds:
                            expression(bound, guard)
                        # DO assigns its control variable even for a zero-trip
                        # loop. It can be a dummy or module variable, rather
                        # than a private local used only for addressing.
                        effect(self._binding(scope, iterator), "write", guard, str(iterator))
                        statements(children[1:-1], guard + (str(header),))
                elif kind == "Associate_Construct":
                    association = self._associate_scopes.get(id(node))
                    if association is None or not association.available:
                        reasons.append(getattr(association, "reason", None) or "unproved lexical ASSOCIATE selector")
                        emit({"kind": "boundary", "source": str(node), "guard": guard, "reason": reasons[-1]})
                    else:
                        statements(node.content[1:-1], guard)
                elif kind == "Comment":
                    continue
                elif kind in {"Continue_Stmt", "Return_Stmt"}:
                    emit({"kind": "control", "source": str(node), "guard": guard})
                elif kind in {"Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt", "Nullify_Stmt"}:
                    reasons.append(f"storage lifetime/association boundary: {kind}")
                    emit({"kind": "boundary", "source": str(node), "guard": guard, "reason": reasons[-1]})
                else:
                    reasons.append(f"native ordering/effects unavailable: {kind}")
                    emit({"kind": "boundary", "source": str(node), "guard": guard, "reason": reasons[-1]})

        # Specification bounds execute on procedure entry too. Descriptor
        # inquiries remain metadata; indexed payloads need native coherence,
        # and unresolved specification functions make this closure incomplete.
        for declaration in _children(_part(routine.scope.node, "Specification_Part")):
            if _kind(declaration) != "Type_Declaration_Stmt":
                continue
            dtype, attributes, entities = declaration.items
            dimension = next((a.items[1] for a in _children(attributes)
                              if _kind(a) == "Dimension_Attr_Spec"), None)
            for entity in _children(entities):
                shape = entity.items[1] if entity.items[1] is not None else dimension
                if _kind(shape) == "Assumed_Size_Spec":
                    for axis in _children(shape.items[0]):
                        for bound in axis.items:
                            expression(bound, ())
                    expression(shape.items[1], ())
                else:
                    for axis in _children(shape):
                        for bound in axis.items:
                            expression(bound, ())
            if _kind(dtype) == "Intrinsic_Type_Spec" and str(dtype.items[0]).lower() == "character":
                expression(dtype.items[1], ())

        statements(_children(routine.execution))
        if not any(operation["kind"] in {"call","native_contract"} for operation in operations):
            # A leaf read before any payload write cannot consume values supplied
            # by its caller through INTENT(OUT). Do not hide this event by changing
            # the normalized access intent. This is an initial diagnostic, not a
            # complete proof for reads after partial or conditional writes.
            written = set()
            for operation in operations:
                root = operation.get("resource")
                if (operation["kind"] == "read" and operation["rank"]
                        and root in summary["definition_changes"] and root not in written):
                    diagnostic = {"procedure":requested,"resource":root,"source_access":operation["source_access"],
                                  "reason":"INTENT(OUT) payload read before any source write"}
                    if diagnostic not in summary["definition_diagnostics"]:
                        summary["definition_diagnostics"].append(diagnostic)
                elif operation["kind"] in {"write","overwrite"}:
                    written.add(root)
        summary["reasons"] = list(dict.fromkeys(reasons))
        summary["capture_lifetime_requirements"] = [lifetime_requirements[root] for root in sorted(lifetime_requirements)]
        summary["complete"] = not reasons
        summary["guaranteed_whole_overwrites"] = self.whole_overwrites(routine) if summary["complete"] else []
        summary["ordered_effects"] = self._compose_effects(summary, _closure)
        if not summary["complete"]:
            summary["guaranteed_whole_overwrites"] = []
        summary["effect_composition"] = {
            "available": summary["complete"], "operations": len(summary["ordered_effects"]),
            "coordinate_system": "canonical roots with original logical view chains; physical mapping requires descriptors",
            "reason": None if summary["complete"] else "source proof or bounded effect composition incomplete"}
        directives = [str(node) for node in walk(routine.scope.node)
                      if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")]
        summary["openmp_directives"] = directives
        summary["native_completion"] = self._native_completion(routine, summary, _closure)
        summary["cloneable"] = (routine.source_kind == "module" and summary["complete"] and not persistent
                                and not directives and not any(effect["kind"] in {"environment_read", "environment_write"}
                                                              for effect in summary["ordered_effects"]))
        sections = self.native_sections(requested)
        summary["native_sections"] = sections.public()
        summary["section_precision"] = ("typed bounded native rectangles; checked physical mapping required"
                                        if sections.available else "whole-resource conservative effects")
        return self._stamp_summary(summary)

    def _compose_effects(self, summary, closure):
        """Project ordered call effects without inventing a physical rectangle.

        A whole formal overwrite through a section or an explicit-shape dummy
        covers that view only. Retaining the typed view chain also preserves
        each procedure's original coordinates and descriptor dependencies.
        Expansion has the same finite operation budget as source analysis.
        """
        if not summary["complete"]:
            return []
        result = []
        from compiler.frontend.structured_effects import mapped_resource

        def append(item):
            if len(result) >= self.operation_limit:
                reason = "composed source operation budget exhausted"
                if reason not in summary["reasons"]:
                    summary["reasons"].append(reason)
                summary["complete"] = False
                return False
            result.append(item)
            return True

        def frame(operation):
            return [{"procedure": summary["procedure"], "condition": condition}
                    for condition in operation.get("guard", ())]

        for resource in summary["definition_changes"]:
            if not append({"kind": "definition_change", "resource": resource,
                           "coverage": "whole_formal", "source_procedure": summary["procedure"],
                           "position": "procedure entry under original presence and allocation semantics",
                           "view_chain": [], "guard_frames": []}):
                return result
        for operation in summary["operations"]:
            if operation["kind"] == "call":
                child = closure.summaries[operation["procedure"]]
                mappings = {item["formal_resource"]: item for item in operation["resource_mappings"]}
                for effect in child["ordered_effects"]:
                    effect = deepcopy(effect)
                    mapped = mappings.get(effect.get("resource"))
                    if mapped is None:
                        parent = next((item for root, item in sorted(mappings.items(), key=lambda pair: -len(pair[0]))
                                       if effect.get("resource", "").startswith(root + "%")), None)
                        if parent is not None:
                            projected = mapped_resource(effect["resource"],
                                {parent["formal_resource"]: parent["resource"]})
                            mapped = {**parent, "resource": projected,
                                      "component_suffix": effect["resource"][len(parent["formal_resource"]):]}
                    if mapped is not None:
                        if mapped["resource"] is None:
                            # Omitted optional storage and scalar expressions
                            # have no caller array root. Actual-expression
                            # reads were already emitted at the original call.
                            continue
                        effect["resource"] = mapped["resource"]
                        effect["view_chain"] = [{"caller": summary["procedure"],
                            "callee": operation["procedure"], **deepcopy(mapped)},
                            *effect.get("view_chain", ())]
                    effect["guard_frames"] = [*frame(operation), *effect.get("guard_frames", ())]
                    if not append(effect):
                        return result
            elif operation["kind"] == "native_contract":
                for effect in operation["effects"]:
                    if not append({**deepcopy(effect), "source_procedure": operation["procedure"],
                        "contract_identity": operation["identity"], "contract_sha256": operation["contract_sha256"],
                        "view_chain": [], "guard_frames": frame(operation),
                        "coverage": "whole_contracted_resource"}):
                        return result
            elif operation["kind"] == "native_environment":
                for effect in operation["effects"]:
                    if not append({**deepcopy(effect), "source_procedure": summary["procedure"],
                                   "native_environment_identity": operation["proof_identity"],
                                   "view_chain": [], "guard_frames": frame(operation)}):
                        return result
            elif operation["kind"] in {"read", "write", "overwrite", "descriptor_read"}:
                if not append({**deepcopy(operation), "source_procedure": summary["procedure"],
                    "view_chain": [], "guard_frames": frame(operation),
                    "coverage": "whole_formal" if operation["kind"] == "overwrite" else "conservative_resource_effect"}):
                    return result
        return result

    def _native_completion(self, routine, summary, closure):
        """Prove only source-backed synchronous worksharing for serial owners.

        This does not rewrite directives, prove parallel independence, or refine
        array sections. The original compiler/OpenMP runtime still executes the
        helper; its implicit completion must precede our host coherence hook.
        """
        children = [closure.summaries.get(op["procedure"], {}) for op in summary["operations"]
                    if op["kind"] == "call"]
        modules = {routine.scope.module} if routine.scope.module in self.modules else set()
        for operation in summary["operations"]:
            resources = [operation.get("resource", ""), *operation.get("resource_mapping", {}).values()]
            resources.extend(effect.get("resource", "") for effect in operation.get("effects", ()))
            modules.update(resource.split("::", 1)[0] for resource in resources
                           if resource.split("::", 1)[0] in self.modules)
        module_directives = [str(node) for module in sorted(modules)
                             for node in walk(_part(self.modules[module].node, "Specification_Part"))
                             if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")]
        has_directives = bool(summary["openmp_directives"] or module_directives) or any(
            child.get("native_completion", {}).get("has_openmp_in_closure", True) for child in children)
        has_opaque_calls = any(operation["kind"] == "native_contract" for operation in summary["operations"]) or any(
            child.get("native_completion", {}).get("has_opaque_calls_in_closure", True) for child in children)
        result = {"available": False, "reason": None, "caller_contract": "serial_source_scope",
                  "requires_serial_caller": True, "has_openmp_in_closure": has_directives,
                  "has_opaque_calls_in_closure": has_opaque_calls}

        def boundary(reason):
            result["reason"] = reason
            return result

        if not summary["complete"]:
            return boundary("native source effects are incomplete")
        if module_directives:
            return boundary("enclosing module OpenMP ownership is unproven: " + module_directives[0])
        for child in children:
            if not child.get("native_completion", {}).get("available", False):
                return boundary("native callee completion is unproven: " + child.get("procedure", "unknown"))

        nodes = [node for node in walk(routine.scope.node)
                 if _kind(node) == "Comment" or _kind(node).endswith("_Stmt")]
        positions = {id(node): index for index, node in enumerate(nodes)}
        loops = {}
        for node in walk(routine.execution):
            if _kind(node) not in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}:
                continue
            body = [item for item in _children(node) if _kind(item) != "Comment"]
            if body and _kind(body[-1]) == "End_Do_Stmt":
                loops[id(body[0])] = body[-1]

        def directive(node):
            text = str(node).lstrip().lower()
            return text[5:].split() if _kind(node) == "Comment" and text.startswith("!$omp") else None

        def next_statement(index):
            while index < len(nodes) and _kind(nodes[index]) == "Comment" and directive(nodes[index]) is None:
                index += 1
            return index

        associated = set()
        index = 0
        while index < len(nodes):
            tokens = directive(nodes[index])
            if tokens is None:
                index += 1
                continue
            if tokens not in (["do"], ["parallel", "do"]):
                return boundary("unsupported native OpenMP directive: " + str(nodes[index]))
            header_index = next_statement(index + 1)
            if header_index == len(nodes) or id(nodes[header_index]) not in loops:
                return boundary("native OpenMP directive is not associated with a complete DO loop")
            end_index = positions[id(loops[id(nodes[header_index])])]
            if any(directive(node) is not None for node in nodes[header_index:end_index + 1]):
                return boundary("nested native OpenMP directives are unsupported")
            close_index = next_statement(end_index + 1)
            if close_index == len(nodes) or directive(nodes[close_index]) != ["end", *tokens]:
                return boundary("native OpenMP loop requires a matching clause-free end directive")
            associated.add(str(nodes[header_index]))
            index = close_index + 1

        # A called helper inside a worksharing loop must not introduce a hidden
        # parallel/worksharing construct. Opaque serial contracts do not prove
        # this source-level absence of nested teams.
        for operation in summary["operations"]:
            if not associated.intersection(operation.get("guard", ())):
                continue
            if operation["kind"] == "native_contract":
                return boundary("opaque native call inside OpenMP loop has no nested-team proof")
            if operation["kind"] == "call":
                child = closure.summaries.get(operation["procedure"], {})
                if child.get("native_completion", {}).get("has_openmp_in_closure", True):
                    return boundary("native OpenMP loop calls a helper with OpenMP directives: " + operation["procedure"])
                if child.get("native_completion", {}).get("has_opaque_calls_in_closure", True):
                    return boundary("native OpenMP loop calls a helper with opaque nested-team behavior: " + operation["procedure"])

        result["available"] = True
        result["reason"] = ("matched clause-free OpenMP loops complete for a serial source caller"
                            if has_directives else "source closure has no OpenMP directives")
        return result

    def native_sections(self, requested):
        """Return typed original references without parsing explanatory strings."""
        self._analysis_started = True
        self.inputs.verify()
        if requested not in self.routines:
            raise CompilationError("source procedure unavailable: " + requested)
        authority = sha256(_canonical(self._summary_authority()).encode()).hexdigest()
        key = requested, authority
        if key not in self._native_sections:
            result = analyze_native_sections(self, self.routines[requested])
            if len(self._native_sections) >= self.procedure_limit:
                self._native_sections.pop(next(iter(self._native_sections)))
            self._native_sections[key] = result
        return self._native_sections[key]

    def source_scope_for(self, node, default=None):
        from compiler.frontend.component_bindings import source_scope_for
        return source_scope_for(self, node, default)

    def _require_original(self, requested):
        """Only original parsed source, not lookalike ASTs, grants authority."""
        self.inputs.verify()
        routine = self.routines.get(requested)
        original = self._source_roles.get(requested)
        numerical = self._numerical_roles.get(requested)
        if any(getattr(node, "item", None) is not item or item.span != span
               or getattr(item, "fort_original_span", None) != original_span
               for node, item, span, original_span in self._source_provenance.values()):
            raise CompilationError("structured effects require unchanged original source span authority")
        if any(name not in self.modules or self.modules[name].node is not node or str(node) != text
               for name, (node, text) in getattr(self, "_source_module_roles", {}).items()):
            raise CompilationError("structured effects require unchanged original source-backed module authority")
        if (routine is None or original is None or routine.execution is not original[0]
                or self._routine_signature(routine) != original[1] or str(routine.execution) != original[2]
                or numerical is None or routine.scope.node is not numerical[0]
                or str(routine.scope.node) != numerical[2]):
            raise CompilationError("structured effects require original source-backed procedure authority")
        return routine

    def structure(self, requested):
        """Return one reusable local skeleton without expanding its callees."""
        from compiler.frontend.structured_effects import admit_cached_structure, build_structure
        self._analysis_started = True
        routine = self._require_original(requested)
        authority = self._summary_authority()
        digest = sha256(_canonical(authority).encode()).hexdigest()
        key = requested, digest
        if key not in self._structures:
            graph = build_structure(self, routine, digest)
            cache_authority = {**authority, "payload_role": "structured_source"}
            cached = self._summary_cache.lookup(cache_authority, requested)
            if cached is not None:
                try:
                    admit_cached_structure(cached["structure"], graph)
                except (KeyError, TypeError, ValueError):
                    self._cache_rejections += 1
                    cached = None
            if cached is None:
                self._summary_cache.store(cache_authority, requested, {"structure": graph.public()})
            if len(self._structures) >= self.procedure_limit:
                self._structures.pop(next(iter(self._structures)))
            self._structures[key] = graph
        return self._structures[key]

    def _selected_source(self, requested, selected):
        graph = self.structure(requested)
        if not graph.available:
            raise CompilationError("structured source skeleton unavailable: " + "; ".join(graph.reasons))
        if isinstance(selected, str) or not isinstance(selected, (tuple, list)):
            selected = (selected,)
        group = graph.native_group_for_selection(selected)
        if group is not None:
            return graph, (group.node_id,)
        identities, original_node_ids = [], None
        for item in selected:
            if _kind(item) == "Comment" and not str(item).lstrip().lower().startswith("!$omp"):
                # Documentation has no execution identity. It may accompany
                # an exact original group, but a foreign comment cannot grant
                # source authority to a projected selection.
                if original_node_ids is None:
                    # structure() authenticated this exact routine above.
                    # One selection may contain many documentation comments;
                    # none needs another global source/AST validation pass.
                    original_node_ids = {id(node) for node in walk(self.routines[requested].execution)}
                if id(item) not in original_node_ids:
                    raise CompilationError("selected comment lacks original source authority")
                continue
            if isinstance(item, str):
                if item not in graph.nodes:
                    raise CompilationError("selected structured effect ID lacks original source authority")
                identity = item
            else:
                matches = [graph._ids[key] for key in ((id(item), "statement"), (id(item), "condition"),
                                                     (id(item), "header"), (id(item), "selector"))
                           if key in graph._ids]
                if len(set(matches)) != 1:
                    raise CompilationError("selected effect node lacks unambiguous original source authority")
                identity = matches[0]
            if identity in identities:
                raise CompilationError("selected source effect node is repeated")
            identities.append(identity)
        if len(identities) > 1:
            positions = {identity: index for index, identity in enumerate(graph.nodes)}
            if len({graph.nodes[identity].guard for identity in identities}) != 1:
                raise CompilationError("selected effects must form one reached guarded segment")
            owned = set()
            for identity in identities:
                subtree = {id(item) for node in graph.source_nodes(identity) for item in walk(node)
                           if hasattr(item, "items") or hasattr(item, "content")}
                if owned & subtree:
                    raise CompilationError("selected source effect subtrees overlap")
                owned.update(subtree)
            if [positions[identity] for identity in identities] != sorted(positions[identity] for identity in identities):
                raise CompilationError("selected effect nodes must retain original source order")
        return graph, tuple(identities)

    def _segment_projection(self, requested, identities, *, include_entry=False, capture_locals=False,
                            _graph=None):
        """Create a private proof projection only after original-node checks."""
        from fparser.two import Fortran2003 as F
        graph = self.structure(requested) if _graph is None else _graph
        # The immediately preceding structure() call already authenticated
        # the original routine, provenance and current analysis authority.
        original = self.routines[requested]
        analysis = copy(self)
        analysis.routines = dict(self.routines)
        analysis.summaries, analysis._closures, analysis._native_sections = {}, {}, {}
        analysis._structures, analysis._segments, analysis._descriptor_proofs = {}, {}, {}
        analysis._descriptor_source = self
        routine = copy(original)
        routine.scope = copy(original.scope)
        routine.scope.bindings = {name: copy(binding) for name, binding in original.scope.bindings.items()}
        if not include_entry:
            for binding in routine.scope.bindings.values():
                binding.intent = None
        analysis._captured_local_roots = frozenset(binding.root for binding in routine.scope.bindings.values()
                                                   if capture_locals and binding.rank)
        body, guards = [], []
        for identity in identities:
            node = graph.nodes[identity]
            originals = graph.source_nodes(identity)
            if node.kind == "entry":
                if not include_entry:
                    raise CompilationError("source entry effects require include_entry=True")
                continue
            if node.details.get("evaluation") == "condition":
                probe = "fort_effect_condition"
                routine.scope.bindings[probe] = Binding(probe, requested + "::" + probe, "logical", 4, 0)
                statement = F.Assignment_Stmt(probe + " = .true.")
                statement.items = (statement.items[0], statement.items[1], originals[0])
                body.append(statement)
            elif node.details.get("evaluation") == "loop_header":
                header = originals[0]
                control = next((item for item in _children(header) if _kind(item) == "Loop_Control"), None)
                if control is None or control.items[1] is None:
                    raise CompilationError("source loop header requires counted DO control")
                iterator, bounds = control.items[1]
                for index, expression in enumerate(bounds):
                    if expression is None:
                        continue
                    probe = "fort_effect_bound_" + str(index)
                    routine.scope.bindings[probe] = Binding(probe, requested + "::" + probe, "integer", 4, 0)
                    statement = F.Assignment_Stmt(probe + " = 0")
                    statement.items = (statement.items[0], statement.items[1], expression)
                    body.append(statement)
                statement = F.Assignment_Stmt(str(iterator) + " = 0")
                body.append(statement)
            elif node.details.get("evaluation") == "associate_selector":
                # Admitted selectors establish variable aliases, not payload
                # values. Their association/descriptor checks remain source
                # prerequisites, without falsely uploading a scalar field.
                body.append(F.Continue_Stmt("continue"))
            else:
                body.extend(originals)
            guards.extend(node.guard)
        routine.execution = copy(original.execution)
        routine.execution.content = body
        routine.scope.node = copy(original.scope.node)
        specification = _part(original.scope.node, "Specification_Part")
        routine.scope.node.content = ([specification] if include_entry and specification is not None else []) + [routine.execution]
        if capture_locals:
            routine.arguments = tuple(name for name, binding in routine.scope.bindings.items()
                                      if not name.startswith("fort_effect_") and "allocatable" not in binding.attributes)
        analysis.routines[requested] = routine
        return analysis, routine, tuple(dict.fromkeys(guards))

    def segment_summary(self, requested, selected, *, include_entry=False, capture_locals=False):
        """Materialize a reached original segment under its own proof budget.

        Call effects still require a complete bounded child proof. A mixed
        coordinator traverses child skeletons instead of treating an incomplete
        child as one opaque native operation. No guard or bound is evaluated.
        """
        graph, identities = self._selected_source(requested, selected)
        if graph.deferred_native_descendants(identities):
            raise CompilationError("deferred native group requires whole original native completion authority")
        key = graph.identity, identities, bool(include_entry), bool(capture_locals)
        if key in self._segments:
            return deepcopy(self._segments[key])
        analysis, _routine, guard = self._segment_projection(requested, identities,
            include_entry=include_entry, capture_locals=capture_locals)
        summary = analysis.summarize(requested)
        summary["structured_identity"] = graph.identity
        summary["selected_node_ids"] = list(identities)
        summary["includes_entry"] = bool(include_entry)
        # Materializing a branch/loop retains its inner original guards. The
        # outer source guards describe where this demanded segment is reached;
        # they are evidence and must never become early runtime predicates.
        for operation in summary["operations"]:
            operation["guard"] = (*guard, *operation.get("guard", ()))
            operation["node_ids"] = list(identities)
        for effect in summary.get("ordered_effects", ()):
            effect["guard_frames"] = [{"procedure": requested, "condition": condition}
                                       for condition in guard] + effect.get("guard_frames", [])
            effect["node_ids"] = list(identities)
        summary["analysis_identity"] = graph.authority_identity
        summary["summary_role"] = "reached_source_segment"
        summary["demand_identity"] = sha256(_canonical({"graph": graph.identity, "nodes": identities,
            "entry": bool(include_entry), "capture_locals": bool(capture_locals)}).encode()).hexdigest()
        summary["summary_identity"] = sha256(_canonical({key: value for key, value in summary.items()
                                                         if key != "summary_identity"}).encode()).hexdigest()
        if len(self._segments) >= self.procedure_limit:
            self._segments.pop(next(iter(self._segments)))
        self._segments[key] = deepcopy(summary)
        return summary

    def native_group_summaries(self, requested, selected, completion):
        """Demand bounded units only after authenticating a complete native group.

        These projections do not grant standalone execution, numerical outlining
        or a cut inside the retained original team. Generic segment consumers
        continue to see an explicit source boundary.
        """
        graph, identities = self._selected_source(requested, selected)
        group = graph.native_group_for_selection(selected)
        if group is None or not group.available:
            raise CompilationError("native group lacks an available exact original source selection")
        from compiler.frontend.native_completion import NativeCompletionProof
        if not isinstance(completion, NativeCompletionProof):
            raise CompilationError("deferred native group requires its registered whole completion proof")
        completion.validate(self, requested, identities)
        summaries = []
        for unit in group.units:
            analysis, _routine, guard = self._segment_projection(requested, unit.selected_node_ids,
                capture_locals=True, _graph=unit.structure)
            summary = analysis.summarize(requested)
            for operation in summary["operations"]:
                operation["guard"] = tuple(dict.fromkeys((*unit.guard, *guard, *operation.get("guard", ()))))
            summary.update(structured_identity=graph.identity, selected_node_ids=list(identities),
                           summary_role="complete_native_group_unit",
                           demand_identity=unit.id, native_group_identity=group.identity)
            summary["summary_identity"] = sha256(_canonical({key: value for key, value in summary.items()
                                                             if key != "summary_identity"}).encode()).hexdigest()
            summaries.append(summary)
        return graph, identities, group, tuple(summaries)

    def joined_completion(self, requested, selected):
        """Issue a compiler-proved token for one whole original joined group."""
        from compiler.frontend.native_completion import prove_joined_completion
        proof = prove_joined_completion(self, requested, selected)
        if proof.identity not in self._joined_completions:
            if len(self._joined_completions) >= self.procedure_limit:
                self._joined_completions.pop(next(iter(self._joined_completions)))
            self._joined_completions[proof.identity] = proof
        return self._joined_completions[proof.identity]

    def native_environment(self, requested, original_call):
        """Authenticate an unchanged native IEEE environment operation."""
        from compiler.frontend.native_environment import prove_native_environment
        return prove_native_environment(self, requested, original_call)

    def native_environment_state(self, requested, binding):
        """Prove invocation-local LOGICAL storage remains exclusively native."""
        from compiler.frontend.native_environment import prove_native_state
        return prove_native_state(self, requested, binding)

    def numerical_joined_completion(self, requested, selected):
        """Prove source-helper completion for numerical outlining only.

        This distinct authority cannot authorize native coherence effects;
        numerical lowering must separately prove the reached computation.
        """
        from compiler.frontend.numerical_completion import prove_numerical_completion
        return prove_numerical_completion(self, requested, selected)

    def worksharing_completion(self, requested, joined, selected):
        """Prove one synchronized DO's participation within its original team."""
        from compiler.frontend.worksharing_completion import prove_worksharing_completion
        proof = prove_worksharing_completion(self, requested, joined, selected)
        if proof.identity not in self._worksharing_completions:
            if len(self._worksharing_completions) >= self.operation_limit:
                self._worksharing_completions.pop(next(iter(self._worksharing_completions)))
            self._worksharing_completions[proof.identity] = proof
        return self._worksharing_completions[proof.identity]

    def worksharing_native_completion(self, requested, joined, selected):
        """Prove native coherence at bounded original-team worksharing cuts.

        This authority retains the original team, work and synchronization. It
        cannot authorize a numerical worker or standalone subunit execution.
        """
        from compiler.frontend.worksharing_completion import prove_worksharing_native_completion
        proof = prove_worksharing_native_completion(self, requested, joined, selected)
        if proof.identity not in self._worksharing_native_completions:
            if len(self._worksharing_native_completions) >= self.operation_limit:
                self._worksharing_native_completions.pop(next(iter(self._worksharing_native_completions)))
            self._worksharing_native_completions[proof.identity] = proof
        return self._worksharing_native_completions[proof.identity]

    def reduction_candidates(self, requested, selected=None, *, limit=32):
        """Lazily prove intrinsic reductions in authenticated reached nodes.

        The local skeleton bounds this scan; direct callees are never expanded.
        Returned source facts do not grant an executable reduction strategy or
        calibrated automatic placement. The original native statement remains
        the implementation until a separately admitted worker is generated.
        """
        from compiler.frontend.reductions import analyze_reduction, candidate_nodes
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise CompilationError("reduction candidate limit must be positive")
        limit = min(limit, 32, self.operation_limit)
        graph = self.structure(requested)
        identities = None if selected is None else self._selected_source(requested, selected)[1]
        key = graph.identity, identities, limit
        if key not in self._reduction_candidates:
            candidates = candidate_nodes(graph, identities) if graph.available else ()
            records = [analyze_reduction(self, requested, identity).public() for identity in candidates[:limit]]
            result = {"schema_version": 1, "procedure": requested,
                      "structured_identity": graph.identity,
                      "source_analysis_available": graph.available,
                      "execution_supported": False,
                      "execution_reason": "source proofs only; generated reduction execution is not implemented",
                      "selected_original_nodes": list(identities) if identities is not None else None,
                      "candidate_limit": limit, "candidate_count": len(candidates),
                      "truncated": len(candidates) > limit,
                      "reason": "; ".join(graph.reasons) if not graph.available else
                          "bounded reduction candidate limit exceeded" if len(candidates) > limit else None,
                      "records": records}
            if len(self._reduction_candidates) >= self.procedure_limit:
                self._reduction_candidates.pop(next(iter(self._reduction_candidates)))
            self._reduction_candidates[key] = deepcopy(result)
        return deepcopy(self._reduction_candidates[key])

    def openmp_reduction(self, requested, selected):
        """Prove one requested original joined reduction group, without lowering."""
        from compiler.frontend.omp_reductions import analyze_openmp_reduction
        return analyze_openmp_reduction(self, requested, selected)

    def native_sections_for_nodes(self, requested, selected, *, include_entry=False, capture_locals=False,
                                 completion=None):
        """Typed original-coordinate refinement for a demanded source segment."""
        graph, identities = self._selected_source(requested, selected)
        if completion is not None:
            from compiler.frontend.native_completion import NativeCompletionProof
            from compiler.frontend.worksharing_completion import WorksharingNativeCompletionProof
            if not isinstance(completion, (NativeCompletionProof, WorksharingNativeCompletionProof)):
                raise CompilationError("native section completion requires a registered original source proof token")
            completion.validate(self, requested, identities)
        group = graph.native_group_for_selection(selected)
        if group is not None:
            return self._native_group_sections(requested, group, completion)
        if graph.deferred_native_descendants(identities):
            raise CompilationError("deferred native sections require the exact whole original source selection")
        analysis, routine, _guard = self._segment_projection(requested, identities,
            include_entry=include_entry, capture_locals=capture_locals)
        # The section analyzer needs original descriptor declarations even
        # though this inner operation does not execute their entry expressions.
        specification = _part(self.routines[requested].scope.node, "Specification_Part")
        routine.scope.node.content = ([specification] if specification is not None else []) + [routine.execution]
        analysis._native_sections_include_entry = bool(include_entry)
        analysis._native_sections_completion = completion
        sections = analyze_native_sections(analysis, routine)
        if completion is not None and sections.available:
            # Counted-loop iterators have already been replaced by their
            # original ranges. Any remaining PRIVATE dependency is per-thread
            # state: a coordinator's value cannot describe the complete team's
            # physical accesses, including state carried from an earlier DO.
            dependencies = {dependency.resource for resource in sections.resources
                            for dependency in resource.dependencies}
            private = dependencies.intersection(completion.private_roots)
            if private:
                from compiler.frontend.native_sections import NativeSections
                return NativeSections(False, 'native section footprint depends on thread-private state: '
                                      + ', '.join(sorted(private)))
        return sections

    def _native_group_sections(self, requested, group, completion):
        """Refine the whole group's union without expanding a flat closure.

        Bounds are evaluated only at the original whole-group boundary. A
        scalar changed anywhere inside the group cannot supply such a bound.
        Unknown or oversized unions retain conservative whole-resource hooks.
        """
        from dataclasses import replace

        from compiler.frontend.native_completion import NativeCompletionProof
        from compiler.frontend.native_sections import RECTANGLE_LIMIT, NativeSections
        if not isinstance(completion, NativeCompletionProof) or not group.available:
            raise CompilationError("native group sections require whole original completion authority")
        resources, scalar_writes = {}, set()

        def box_identity(box):
            return _canonical({key: value for key, value in box.public().items() if key != "source_access"})

        def unique_boxes(boxes):
            return tuple({box_identity(box): box for box in boxes}.values())
        for node in group.original_nodes:
            for statement in walk(node):
                if _kind(statement) != "Assignment_Stmt":
                    continue
                target = statement.items[0]
                binding = self._binding(self.source_scope_for(statement, self.routines[requested].scope),
                                        target.items[0] if _kind(target) == "Part_Ref" else target)
                if binding is not None and not binding.rank:
                    scalar_writes.add(binding.root)
        for unit in group.units:
            analysis, routine, _guard = self._segment_projection(requested, unit.selected_node_ids,
                capture_locals=True, _graph=unit.structure)
            specification = _part(self.routines[requested].scope.node, "Specification_Part")
            routine.scope.node.content = ([specification] if specification is not None else []) + [routine.execution]
            analysis._native_sections_include_entry = False
            analysis._native_sections_completion = completion
            sections = analyze_native_sections(analysis, routine)
            if not sections.available:
                return NativeSections(False, sections.reason)
            for resource in sections.resources:
                dependencies = {dependency.resource for dependency in resource.dependencies}
                if dependencies.intersection(set(completion.private_roots) | scalar_writes):
                    return NativeSections(False, "native group footprint depends on private or internally changed scalar state")
                if unit.guard and any(dependency.kind == "scalar_read" for dependency in resource.dependencies):
                    return NativeSections(False, "guarded native group footprint requires an unproved early scalar bound")
                previous = resources.get(resource.resource)
                if previous is None:
                    # A guarded or repeated native writer never establishes a
                    # whole overwrite or a whole-group definition.
                    resources[resource.resource] = replace(resource,
                        reads=unique_boxes(resource.reads),
                        writes=unique_boxes((*resource.writes, *resource.overwrites)), overwrites=())
                    previous = resources[resource.resource]
                else:
                    if (previous.rank, previous.lower_bounds, previous.lower_bound_expressions,
                        previous.descriptor_lower_bounds, previous.descriptor_origin) != (
                        resource.rank, resource.lower_bounds, resource.lower_bound_expressions,
                        resource.descriptor_lower_bounds, resource.descriptor_origin):
                        return NativeSections(False, "native group resource descriptors are inconsistent")
                    merged = {}
                    for label in ("reads", "writes"):
                        incoming = (*resource.writes, *resource.overwrites) if label == "writes" else resource.reads
                        unique = {box_identity(box): box for box in (*getattr(previous, label), *incoming)}
                        if len(unique) > RECTANGLE_LIMIT:
                            return NativeSections(False, "native group rectangle union exceeds bounded refinement")
                        merged[label] = tuple(unique.values())
                    resources[resource.resource] = replace(previous, **merged)
                resource = resources[resource.resource]
                if (len(resources) > self.operation_limit or
                        len({box_identity(box) for box in (*resource.reads, *resource.writes)}) > RECTANGLE_LIMIT):
                    return NativeSections(False, "native group section union exceeds bounded refinement")
        return NativeSections(True, resources=tuple(resources[key] for key in sorted(resources)))

    def descriptor_stability(self, requested):
        from compiler.frontend.structured_effects import descriptor_stability
        if hasattr(self, "_descriptor_source"):
            return self._descriptor_source.descriptor_stability(requested)
        graph = self.structure(requested)
        if graph.identity not in self._descriptor_proofs:
            result = descriptor_stability(self, requested)
            if len(self._descriptor_proofs) >= self.procedure_limit:
                self._descriptor_proofs.pop(next(iter(self._descriptor_proofs)))
            self._descriptor_proofs[graph.identity] = result
        return deepcopy(self._descriptor_proofs[graph.identity])

    def summarize_span(self, procedures):
        """Prove a candidate span with one budget for its distinct closure."""
        self.inputs.verify()
        closure = _Closure()
        for procedure in procedures:
            summary = self.summarize(procedure, _closure=closure)
            if not summary["complete"]:
                raise CompilationError("span effect closure incomplete: " + "; ".join(summary["reasons"]))
        return {"procedures": len(closure.summaries), "operations": closure.operations,
                "depth": max((s["closure_depth"] for s in closure.summaries.values()), default=0)}

    def report(self, entry, *, materialize=True):
        """Publish native evidence, optionally without requesting a closure.

        The default retains the legacy complete-closure report and its cache
        accounting. Reached execution can publish authenticated local structure
        and already-proved complete leaves/segments without expanding inactive
        calls merely to construct diagnostics.
        """
        entry = entry.lower()
        matches = [p for p in self.routines if p == entry or ("::" not in entry and p.split("::")[-1] == entry)]
        if len(matches) != 1:
            raise CompilationError(f"native effect entry {entry!r} is unavailable or ambiguous")
        if not materialize:
            self.inputs.verify()
            authority = self._summary_authority()
            authority_identity = sha256(_canonical(authority).encode()).hexdigest()
            if authority["role"] == "source":
                local_structure = self.structure(matches[0])
                structured = local_structure.public()
                requested_reductions = [item for key, item in self._reduction_candidates.items()
                                        if key[0] == local_structure.identity]
                requested_omp_reductions = [proof.public() for proof in self._omp_reduction_proofs.values()
                                           if proof.procedure == matches[0]
                                           and proof.structured_identity == local_structure.identity]
            else:
                structured = {"available": False, "reason": "private projection has no original structural authority"}
                requested_reductions, requested_omp_reductions = [], []
            # Every row retains its own complete proof and source/configuration
            # authority. A local graph alone never grants complete effects.
            # Projection summaries are excluded; reached segment records have
            # their own original-graph identity and selected-node evidence.
            proofs, rejections = {}, {}
            for item in self.summaries.values():
                if (item.get("complete") and item.get("summary_role") == "source"
                        and item.get("analysis_identity") == authority_identity
                        and not any(operation["kind"] == "call" for operation in item["operations"])):
                    proofs[item["summary_identity"]] = item
                elif (not item.get("complete") and item.get("summary_role") == "source"
                        and item.get("analysis_identity") == authority_identity):
                    rejections[item["summary_identity"]] = item
            for item in self._segments.values():
                if (item.get("complete") and item.get("summary_role") == "reached_source_segment"
                        and item.get("analysis_identity") == authority_identity):
                    proofs[item["summary_identity"]] = item
                elif (not item.get("complete") and item.get("summary_role") == "reached_source_segment"
                        and item.get("analysis_identity") == authority_identity):
                    rejections[item["summary_identity"]] = item
            records = list(proofs.values())
            cache_stats = self._summary_cache.stats
            cache_delta = {name: cache_stats[name] - self._cache_started.get(name, 0) for name in cache_stats}
            cache_delta["rejected"] = cache_delta.get("rejected", 0) + self._cache_rejections
            graph = {"schema_version": 1, "scope": "already materialized complete proofs only; no entry call closure",
                     "nodes": [{"procedure": item["procedure"], "summary_identity": item["summary_identity"],
                                "summary_role": item["summary_role"], "complete": True} for item in records],
                     "edges": [{"caller": item["procedure"], "callee": operation["procedure"], "operation": index,
                                "summary_identity": operation.get("summary_identity"),
                                "guard": operation["guard"],
                                "resource_mappings": operation.get("resource_mappings", [])}
                               for item in records for index, operation in enumerate(item["operations"])
                               if operation["kind"] == "call"]}
            self.inputs.verify()
            return deepcopy({"schema_version": 2, "summary_version": SOURCE_SUMMARY_VERSION,
                    "entry": matches[0], "complete": False, "closure_complete_available": False,
                    "closure_materialization": "not_requested",
                    "compatibility": "complete describes the unrequested whole-entry closure, not reached eligibility; "
                        "materialize=True retains the legacy complete-closure report and counters",
                    "effect_authority": "individual complete proof records and compiler-owned reached scopes; "
                        "local structure alone does not prove transitive effects",
                    "proof_scope": "already materialized complete source leaves and original reached segments",
                    "sources": self.sources, "procedures": records,
                    "materialized_rejections": [{"procedure": item["procedure"], "complete": False,
                        "summary_identity": item["summary_identity"], "summary_role": item["summary_role"],
                        "reasons": item.get("reasons", []),
                        "selected_original_nodes": item.get("selected_node_ids", []),
                        "observed_operations": len(item["operations"])} for item in rejections.values()],
                    "analysis_sources": self.inputs.public(),
                    "capture_lifetime_authorizations": list(self._allocation_authorizations.values()),
                    "budgets": {"depth": self.depth_limit, "procedures": self.procedure_limit,
                                "operations": self.operation_limit},
                    "summarized_operations": sum(len(item["operations"]) for item in records),
                    "summary_cache": cache_delta, "call_graph": graph, "structured_effects": structured,
                    "source_reductions": {"analysis": "requested" if requested_reductions else "not_requested",
                        "source_analysis_available": bool(requested_reductions) and
                            all(item["source_analysis_available"] for item in requested_reductions),
                        "execution_supported": False,
                        "reason": "generated reduction execution is not implemented" if requested_reductions else
                            "source analysis is lazy; request original reached reduction candidates",
                        "requests": requested_reductions},
                    "openmp_reductions": {"analysis": "requested" if requested_omp_reductions else "not_requested",
                        "source_analysis_available": bool(requested_omp_reductions) and
                            all(item["source_analysis_available"] for item in requested_omp_reductions),
                        "execution_supported": False,
                        "execution_reason": "original source contracts only; generated OpenMP reduction execution is not implemented",
                        "records": requested_omp_reductions},
                    "automatic_scope_available": False,
                    "effect_coordinate_system": "logical source evidence; original reached native sections define physical transfers"})
        summary = self.summarize(matches[0])
        closure = self._closures.get(matches[0])
        if closure is None:
            closure = _Closure()
            summary = self.summarize(matches[0], _closure=closure)
        # Reporting the legacy complete closure must not perform a second
        # cache lookup (its public counters describe that closure). Explicit
        # structure()/segment_summary() requests use the separate graph cache.
        if self._summary_authority()["role"] == "source":
            from compiler.frontend.structured_effects import build_structure
            authority = sha256(_canonical(self._summary_authority()).encode()).hexdigest()
            local_structure = build_structure(self, self._require_original(matches[0]), authority)
            structured = local_structure.public()
            requested_reductions = [item for key, item in self._reduction_candidates.items()
                                    if key[0] == local_structure.identity]
            requested_omp_reductions = [proof.public() for proof in self._omp_reduction_proofs.values()
                                       if proof.procedure == matches[0]
                                       and proof.structured_identity == local_structure.identity]
        else:
            structured = {"available": False, "reason": "private projection has no original structural authority"}
            requested_reductions = []
            requested_omp_reductions = []
        self.inputs.verify()
        cache_stats = self._summary_cache.stats
        cache_delta = {name: cache_stats[name] - self._cache_started.get(name, 0) for name in cache_stats}
        cache_delta["rejected"] = cache_delta.get("rejected", 0) + self._cache_rejections
        graph = {"schema_version": 1,
                 "nodes": [{"procedure": item["procedure"], "summary_identity": item["summary_identity"],
                            "complete": item["complete"]} for item in closure.summaries.values()],
                 "edges": [{"caller": name, "callee": operation["procedure"], "operation": index,
                            "summary_identity": operation.get("summary_identity"),
                            "guard": operation["guard"],
                            "resource_mappings": operation.get("resource_mappings", [])}
                           for name, item in closure.summaries.items()
                           for index, operation in enumerate(item["operations"]) if operation["kind"] == "call"]}
        return deepcopy({"schema_version": 1, "summary_version": SOURCE_SUMMARY_VERSION,
                "entry": matches[0], "complete": summary["complete"],
                "sources": self.sources, "procedures": list(closure.summaries.values()),
                "analysis_sources": self.inputs.public(),
                "capture_lifetime_authorizations": list(self._allocation_authorizations.values()),
                "budgets": {"depth": self.depth_limit, "procedures": self.procedure_limit,
                            "operations": self.operation_limit}, "summarized_operations": closure.operations,
                "summary_cache": cache_delta,
                "call_graph": graph,
                "structured_effects": structured,
                "source_reductions": {"analysis": "requested" if requested_reductions else "not_requested",
                    "source_analysis_available": bool(requested_reductions) and
                        all(item["source_analysis_available"] for item in requested_reductions),
                    "execution_supported": False,
                    "reason": "generated reduction execution is not implemented" if requested_reductions else
                        "source analysis is lazy; request original reached reduction candidates",
                    "requests": requested_reductions},
                "openmp_reductions": {"analysis": "requested" if requested_omp_reductions else "not_requested",
                    "source_analysis_available": bool(requested_omp_reductions) and
                        all(item["source_analysis_available"] for item in requested_omp_reductions),
                    "execution_supported": False,
                    "execution_reason": "original source contracts only; generated OpenMP reduction execution is not implemented",
                    "records": requested_omp_reductions},
                "automatic_scope_available": False,
                "effect_coordinate_system": "logical source evidence; bounded call rectangles and conservative leaf effects"})


def analyze_source_effects(paths, entry, *, contracts=None, **budgets):
    """Return public native-effect JSON facts without requiring GPU eligibility."""
    return SourceEffects(paths, contracts=contracts, **budgets).report(entry)
