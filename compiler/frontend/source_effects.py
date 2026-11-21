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
from pathlib import Path
from weakref import WeakValueDictionary

from fparser.two.utils import Base, walk

from compiler.frontend.lowering import _intrinsic_kind_imports, _KindScope
from compiler.frontend.source_inputs import SourceInputs
from compiler.frontend.source_syntax import SourceSyntaxSnapshot
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError, SourceLocation

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




class SourceEffects:
    def __init__(self, paths, *, contracts=None, depth=8, procedures=None, operations=256, analysis_sources=None,
                 summary_cache=None, program_roots=()):
        if min(depth, operations) < 1 or (procedures is not None and procedures < 1):
            raise CompilationError("source effect budgets must be positive")
        self.depth_limit, self.procedure_limit, self.operation_limit = depth, procedures if procedures is not None else 1, operations
        self.modules, self.routines, self.summaries = {}, {}, {}
        self.functions = set()
        self.numerical_helpers = {}
        self._closures = {}
        self._native_sections = {}
        self._structures, self._segments, self._descriptor_proofs = {}, {}, {}
        # Active source contracts retain exact graph objects. Bounded strong
        # cache eviction must not mint a second identity for those contracts;
        # unreferenced graphs can still disappear with their cache entries.
        self._live_structures = WeakValueDictionary()
        self._joined_completions = {}
        self._native_environments, self._native_environment_states = {}, {}
        self._native_predicates = {}
        self._native_functions = {}
        self._descriptor_functions = {}
        self._numerical_completions = {}
        self._worksharing_completions = {}
        self._worksharing_native_completions = {}
        self._source_scopes, self._associate_scopes, self._block_scopes = {}, {}, {}
        self._block_scope_roles = {}
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
        if (type(program_roots) is not tuple or any(type(name) is not str or not name.startswith("$program::")
                or not re.fullmatch(r"[a-z][a-z0-9_]*", name.removeprefix("$program::")) for name in program_roots)):
            raise CompilationError("program metadata requires exact named original program roots")
        external_nodes, program_nodes = [], []
        source_nodes = 0
        for filename in self.inputs.paths:
            tree = self.inputs.parse(filename)
            for node in _children(tree):
                if (_kind(node) == "Main_Program" and _part(node, "Program_Stmt") is not None
                        and "$program::" + str(_part(node, "Program_Stmt").items[1]).lower() in program_roots):
                    program_nodes.append((filename, node))
                elif _kind(node) == "Subroutine_Subprogram":
                    external_nodes.append((filename, node))
                elif _kind(node) == "Function_Subprogram":
                    name = "$external::" + str(_part(node, "Function_Stmt").items[1]).lower()
                    if name in self.functions:
                        raise CompilationError("duplicate external source procedure " + name)
                    self.functions.add(name)
            for node in walk(tree):
                source_nodes += isinstance(node, Base)
                if _kind(node) != "Module":
                    continue
                module = str(_part(node, "Module_Stmt").items[1]).lower()
                if module in self.modules:
                    raise CompilationError(f"duplicate source module {module}")
                self.modules[module] = Scope(module, filename, node)
        if procedures is None:
            # One bound covers all original callees, declarations and exact
            # factory callsites, including unregistered external/program
            # units. Reuse the complete parser walk already required above.
            # Explicit procedure caps remain independent diagnostic limits.
            self.procedure_limit = max(1, source_nodes)
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
        for filename, node in program_nodes:
            self._register_program(filename, node)
        # Declarations/imports are now complete. These lookup results depend
        # only on that graph, never on the kind values populated below. Keep
        # them local to this pass; later lexical scopes start fresh.
        kind_import_lookups = ({}, {}, {}, {}, {})
        try:
            for scope in self.modules.values():
                self._import_kinds(scope, _lookups=kind_import_lookups)
            for routine in self.routines.values():
                self._import_kinds(routine.scope, _lookups=kind_import_lookups)
            for routine in self.numerical_helpers.values():
                self._import_kinds(routine.scope, _lookups=kind_import_lookups)
        finally:
            for lookup in kind_import_lookups:
                lookup.clear()
        for binding, scope, selector in self.kind_expressions:
            # An unresolved kind cannot select a generic overload.
            try:
                binding.kind = scope.kinds.integer(selector, SourceLocation(str(scope.path)))
            except CompilationError:
                # The initial local specification may have resolved a direct
                # intrinsic export before the complete module import graph
                # exposed a conflict. Never retain that provisional width.
                binding.kind = None
        self._source_roles = {name: (routine.execution, self._routine_signature(routine), str(routine.execution))
                              for name, routine in self.routines.items()}
        self._numerical_roles = {name: (routine.scope.node, self._routine_signature(routine), str(routine.scope.node))
                                 for name, routine in self.numerical_helpers.items()}
        from compiler.frontend.component_bindings import register_associates
        for routine in self.numerical_helpers.values():
            register_associates(self, routine)
        self._source_module_roles = {name: (scope.node, str(scope.node)) for name, scope in self.modules.items()}
        self._source_module_bindings = {binding.root: (binding, _canonical(binding.public()))
                                        for scope in self.modules.values() for binding in scope.bindings.values()}
        from compiler.frontend.source_objects import register_source_object_authority
        register_source_object_authority(self)
        from compiler.frontend.source_functions import register_source_function_authority
        register_source_function_authority(self)
        self._source_provenance = {}
        self._source_object_calls = {}
        for routine in self.numerical_helpers.values():
            for ordinal, node in enumerate(walk(routine.scope.node)):
                if _kind(node) == "Call_Stmt":
                    self._source_object_calls[id(node)] = (node, str(node), routine.scope, ordinal)
                item = getattr(node, "item", None)
                if item is not None:
                    self._source_provenance[id(node)] = (node, item, item.span,
                                                         getattr(item, "fort_original_span", None))
        self._source_syntax = SourceSyntaxSnapshot(
            [*(scope.node for scope in self.modules.values()),
             *(routine.scope.node for routine in self.numerical_helpers.values())])



    def verify_source(self):
        """Check that the source files still hold the text that was parsed."""
        self.inputs.verify()



    def private_projection(self):
        """A copy whose own attributes may be set; it shares the declarations and the syntax."""
        result = copy(self)
        result._local_allocation_proofs = {}
        result._local_allocation_context = None
        result._original_sequence_exits = {}
        result._original_reached_positions = {}
        result._original_reached_positions_owner = None
        result._library_storage_calls = {}
        result._library_storage_calls_owner = None
        result._opaque_library_state_proofs = {}
        result._metadata_constructor_proofs = {}
        result._entry_preservations = {}
        result._entry_preservations_owner = None
        result._descriptor_blocks = {}
        result._descriptor_blocks_owner = None
        result._descriptor_team_proofs = {}
        result._descriptor_terminating_calls = {}
        result._owned_record_regions = {}
        result._owned_record_regions_owner = None
        result._original_worksharing_storage = {}
        result._original_worksharing_storage_owner = None
        result._original_worksharing_storage_receipts = {}
        result._descriptor_functions = dict(self._descriptor_functions)
        return result



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

    def _register_program(self, filename, node):
        """Retain an original named main frame for metadata and call ownership."""
        statement = _part(node, "Program_Stmt")
        ending = _part(node, "End_Program_Stmt")
        # Optional ownership metadata must not reject an unrelated legal main
        # frame that this first named-program slice cannot authenticate.
        if statement is None or ending is None:
            return
        name = str(statement.items[1]).lower()
        if ending.items[1] is not None and str(ending.items[1]).lower() != name:
            return
        _attach_execution_directives(node)
        qualified = "$program::" + name
        if qualified in self.routines or qualified in self.functions:
            raise CompilationError("duplicate original program source " + qualified)
        scope = Scope("$program", filename, node)
        scope.qualified = qualified
        issues = self._specification(scope, (), qualified)
        # Main-program variables have SAVE duration; this supplies neither
        # allocation presence nor an initial payload definition.
        for binding in scope.bindings.values():
            if "parameter" not in binding.attributes:
                binding.attributes |= {"save"}
        issues.append("original program frame supplies metadata/ownership only; no numerical or execution authority")
        routine = Routine(scope, (), _part(node, "Execution_Part"), qualified, issues, "program")
        self.routines[qualified] = routine
        self.numerical_helpers[qualified] = routine
        self._register_internal(scope, node)

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
            binding = Binding(result, qualified + "::" + result, base, width, 0, declaring_scope=child)
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

    def _routine_signature(self, routine):
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







    def _import_kinds(self, scope, active=frozenset(), *, _lookups=None):
        if _lookups is None:
            lookups = ({}, {}, {}, {}, {})
            try:
                return self._import_kinds(scope, active, _lookups=lookups)
            finally:
                for lookup in lookups:
                    lookup.clear()
        if id(scope) in active:
            return
        active = active | {id(scope)}
        imported = []
        # No completed-scope/kind-value memo: recursive imports retain their
        # original order and active-cycle behavior. Only immutable declaration
        # lookups are shared within this synchronous import transaction.
        direct_cache, export_cache, intrinsic_cache, binding_cache, selector_cache = _lookups
        lookups = direct_cache, export_cache, intrinsic_cache, binding_cache, selector_cache

        def selectors(target):
            # Index declaration ownership once, without evaluating or walking
            # another scope's expressions. Name traversal stays lazy and in
            # the original first-demand order for each exact owner.
            if None not in selector_cache:
                owners = {}
                for _binding, owner, expression in self.kind_expressions:
                    owners.setdefault(id(owner), (owner, []))[1].append(expression)
                selector_cache[None] = owners
            key = id(target)
            if key not in selector_cache:
                owner = selector_cache[None].get(key)
                expressions = () if owner is None else owner[1]
                names = frozenset(str(node).lower() for expression in expressions
                                  for node in walk(expression) if _kind(node) == "Name")
                selector_cache[key] = target, names
            return selector_cache[key][1]

        def root_binding(target, name):
            # Only complete roots (empty visited set) are reusable. Recursive
            # _binding frames can depend on the active declaration path.
            key = id(target), name
            if key not in binding_cache:
                binding_cache[key] = target, self._binding(target, name)
            return binding_cache[key][1]

        def direct_kinds(target):
            key = id(target)
            if key in direct_cache:
                return direct_cache[key][1]
            result = {}
            for node in _children(_part(target.node, "Specification_Part")):
                for name, identity in _intrinsic_kind_imports(node).items():
                    result.setdefault(name, set()).add(identity)
            direct_cache[key] = target, result
            return result

        def intrinsic_export(start, name):
            lookup_key = id(start), name
            if lookup_key in intrinsic_cache:
                return intrinsic_cache[lookup_key][1]
            # Intrinsic kinds have no ordinary storage Binding. Follow their
            # exact original module exports without inventing a PARAMETER or
            # selecting one of several distinct/unknown declarations.
            def resolve(owner, spelling):
                binding = owner.bindings.get(spelling)
                if binding is not None:
                    return {("binding", id(binding))}, False
                if (spelling in owner.ambiguous_imports or spelling in owner.procedures or spelling in owner.generics
                        or spelling in owner.externals or spelling in owner.procedure_arguments):
                    return set(), True
                identities = {("intrinsic", *item) for item in direct_kinds(owner).get(spelling, ())}
                blocked = False
                explicit = owner.imports.get(spelling)
                sources = ([explicit] if explicit is not None else []) + [
                    (module, spelling) for module in owner.wildcards
                    if spelling not in owner.wildcard_exclusions.get(module, ())]
                for module, remote in sources:
                    target = self.modules.get(module)
                    if target is None:
                        blocked = True
                    elif self._exported(target, remote):
                        child, invalid = yield target, remote
                        identities.update(child)
                        blocked |= invalid
                    elif explicit == (module, remote):
                        blocked = True
                # A USE-associated entity shadows a host-associated entity.
                # Consult the host only after proving no local entity exists.
                if not identities and not blocked and owner.parent is not None:
                    return (yield owner.parent, spelling)
                return identities, blocked

            key = (id(start), name)
            frames, active, completed, returned = [(key, resolve(start, name))], {key}, {}, None
            while frames:
                current, frame = frames[-1]
                try:
                    owner, spelling = frame.send(returned)
                except StopIteration as done:
                    returned = completed[current] = done.value
                    active.remove(current)
                    frames.pop()
                    continue
                child = (id(owner), spelling)
                if child in active:
                    returned = set(), True
                elif child in completed:
                    returned = completed[child]
                else:
                    active.add(child)
                    frames.append((child, resolve(owner, spelling)))
                    returned = None
            identities, blocked = returned
            if blocked or len(identities) > 1:
                result = None, True
            else:
                identity = next(iter(identities), ())
                result = (identity[3] if identity and identity[0] == "intrinsic" else None), False
            # Save only the final root result. DFS subframes can depend on
            # active ancestors; sharing them would change cycle rejection.
            intrinsic_cache[lookup_key] = start, result
            return result

        def exports(module, visited=frozenset()):
            lookup_key = module, visited
            if lookup_key in export_cache:
                return export_cache[lookup_key]
            target = self.modules.get(module)
            if target is None or module in visited:
                return set()
            names = set(target.bindings) | set(target.imports) | set(direct_kinds(target))
            for parent in target.wildcards:
                names.update(exports(parent, visited | {module}))
            result = {name for name in names if self._exported(target, name)}
            # Cyclic/diamond export paths can have many distinct visited
            # sets. This is an optimization budget, never an analysis limit.
            if len(export_cache) < 4096:
                export_cache[lookup_key] = result
            return result

        for local, (module, remote) in scope.imports.items():
            imported.append((local, module, remote))
        for module in scope.wildcards:
            target = self.modules.get(module)
            if target:
                imported += [(name, module, name) for name in sorted(exports(module))]
        candidates = set(direct_kinds(scope)) | selectors(scope)
        for local, module, remote in [*((name, None, None) for name in sorted(candidates)), *imported]:
            if module is not None:
                target = self.modules.get(module)
                if target is None or not self._exported(target,remote):
                    continue
            # Resolve the complete local import set. Choosing one declaration
            # here would silently price/lower an ambiguous imported kind.
            intrinsic, conflict = intrinsic_export(scope, local)
            if conflict:
                if local not in scope.bindings:
                    scope.kinds.values[local] = None
                continue
            binding = root_binding(scope, local)
            if binding is None:
                if intrinsic is not None:
                    scope.kinds.values[local] = intrinsic
                continue
            if binding.dtype != "integer" or binding.rank or "parameter" not in binding.attributes:
                continue
            owner = self.modules.get(binding.root.split("::",1)[0])
            if owner is None:
                continue
            self._import_kinds(owner, active, _lookups=lookups)
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
                abstract = start.items[0] == "ABSTRACT"
                generic = str(start.items[0]).lower() if start.items[0] is not None and not abstract else None
                members = []
                for item in _children(node):
                    if _kind(item) == "Procedure_Stmt":
                        members += [(scope.module, str(p).lower()) for p in item.items[0].items]
                if abstract and all(_kind(item) in {
                        "Interface_Stmt", "End_Interface_Stmt", "Subroutine_Body", "Function_Body", "Comment"}
                        for item in _children(node)):
                    # An abstract interface executes nothing and supplies no
                    # callable implementation. Keep its identifiers in the
                    # existing unresolved-procedure namespace: they shadow
                    # intrinsics/source targets, including through USE renames,
                    # without granting direct or dummy-procedure dispatch.
                    for item in _children(node):
                        header = (_part(item, "Subroutine_Stmt") if _kind(item) == "Subroutine_Body"
                                  else _part(item, "Function_Stmt") if _kind(item) == "Function_Body" else None)
                        if header is not None:
                            scope.procedure_arguments.add(str(header.items[1]).lower())
                elif generic and members:
                    # Separate original interface blocks contribute to one
                    # generic; later blocks must not hide earlier specifics.
                    scope.generics.setdefault(generic, []).extend(members)
                elif generic is None and all(_kind(item) in {
                        "Interface_Stmt", "End_Interface_Stmt", "Subroutine_Body", "Function_Body", "Comment"}
                        for item in _children(node)):
                    # An explicit external declaration executes nothing. Keep
                    # its names as external/shadowing bindings; each call still
                    # needs independent source/effect/interface authority. An
                    # unrelated opaque declaration must not poison the effects
                    # of numerical work elsewhere in this original routine.
                    for item in _children(node):
                        header = (_part(item, "Subroutine_Stmt") if _kind(item) == "Subroutine_Body"
                                  else _part(item, "Function_Stmt") if _kind(item) == "Function_Body" else None)
                        if header is not None:
                            scope.externals.add(str(header.items[1]).lower())
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
        if name in scope.bindings:
            return []
        if name in scope.procedure_arguments:
            # Propagate a noncallable/unresolved declaration through USE and
            # reexports. An empty result would permit a caller to fall back to
            # a same-spelling host or external implementation accidentally.
            return ["$unresolved_procedure::" + getattr(scope, "qualified", scope.module) + "::" + name]
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
        if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/", "**"}:
            from compiler.frontend.source_scalar_types import scalar_numeric_result

            left = self._signature(scope, items[0], _depth + 1)
            right = self._signature(scope, items[2], _depth + 1)
            return scalar_numeric_result(str(items[1]), left, right)
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







    def source_scope_for(self, node, default=None):
        from compiler.frontend.component_bindings import source_scope_for
        return source_scope_for(self, node, default)


    def _require_original_global(self, *, _syntax=None):
        """Check shared original files, provenance and module syntax freshly."""
        self.verify_source()
        syntax = _syntax if _syntax is not None else self._source_syntax.checkpoint()
        if any(getattr(node, "item", None) is not item or item.span != span
               or getattr(item, "fort_original_span", None) != original_span
               for node, item, span, original_span in self._source_provenance.values()):
            raise CompilationError("structured effects require unchanged original source span authority")
        if any(name not in self.modules or self.modules[name].node is not node or not syntax.unchanged(node)
               for name, (node, _text) in getattr(self, "_source_module_roles", {}).items()):
            raise CompilationError("structured effects require unchanged original source-backed module authority")
        return syntax

    def _require_original_routine(self, requested, *, _syntax):
        """Check one exact routine inside an immediate authenticated demand."""
        routine = self.routines.get(requested)
        original = self._source_roles.get(requested)
        numerical = self._numerical_roles.get(requested)
        if (routine is None or original is None or routine.execution is not original[0]
                or self._routine_signature(routine) != original[1] or not _syntax.unchanged(routine.execution)
                or numerical is None or routine.scope.node is not numerical[0]
                or not _syntax.unchanged(routine.scope.node)):
            raise CompilationError("structured effects require original source-backed procedure authority")
        return routine

    def _require_original(self, requested, *, _syntax=None):
        """Only original parsed source, not lookalike ASTs, grants authority."""
        syntax = self._require_original_global(_syntax=_syntax)
        return self._require_original_routine(requested, _syntax=syntax)





















