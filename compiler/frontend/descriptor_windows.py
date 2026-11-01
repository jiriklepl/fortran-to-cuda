"""Original interval descriptor proofs, independent of entry payload facts.

No projected/synthetic body or whole-procedure descriptor proof grants this
authority. Selected original source and its complete direct call closure are
checked; excluded prefixes/tails, reachability, aliases and definitions remain
separate obligations before any owner may open.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields
from hashlib import sha256
from types import GetSetDescriptorType, MappingProxyType, MemberDescriptorType

from fparser.two.utils import Base

from compiler.frontend.call_bindings import resolve_original_call, resolve_source_call
from compiler.frontend.component_bindings import ComponentStorageUnsupported, component_access
from compiler.frontend.descriptor_budget import DescriptorSourceBudget
from compiler.frontend.descriptor_teams import DescriptorTeamProof, prove_descriptor_team
from compiler.frontend.local_allocatable_storage import original_local_allocation
from compiler.frontend.native_completion import NativeCompletionProof
from compiler.frontend.native_environment import NativeEnvironmentProof, NativeLocalStateProof
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.native_intrinsics import NativePredicateProof
from compiler.frontend.native_prefix import NativePrefixProof
from compiler.frontend.numerical_completion import NumericalCompletionProof
from compiler.frontend.source_functions import (
    _scope_roles,
    _validate_resolution,
    _validate_scope,
    _validate_scope_roles,
    _walk,
)
from compiler.frontend.source_objects import _canonical, _members, _TypeBuilder
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

DESCRIPTOR_WINDOW_VERSION = 6
DESCRIPTOR_WINDOW_SOURCE_ACCOUNTING_VERSION = 7
DESCRIPTOR_WINDOW_INTRINSIC_OUTPUT_VERSION = 8
DESCRIPTOR_WINDOW_STORAGE_FUNCTION_VERSION = 9
DESCRIPTOR_WINDOW_BORROWED_ARRAY_VERSION = 10
DESCRIPTOR_WINDOW_FIXED_MODULE_VERSION = 11
DESCRIPTOR_WINDOW_SCALAR_CLOCK_VERSION = 12
DESCRIPTOR_WINDOW_COMPONENT_READ_VERSION = 17
DESCRIPTOR_WINDOW_MATRIX_INTRINSIC_VERSION = 18
DESCRIPTOR_WINDOW_PRIVATE_ALLOCATION_VERSION = 21
DESCRIPTOR_WINDOW_READONLY_RECORD_VERSION = 28
_UNSAFE = {"pointer", "optional", "volatile", "asynchronous", "value"}
_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "present", "sum", "product", "any", "all", "count", "minval", "maxval", "dot_product"}


class DescriptorWindowUnsupported(CompilationError):
    """An unchanged original operation may run only after its owner closes.

    Original-source/token validation and aggregate analysis-budget failures
    never use this signal. It grants no descriptor or execution authority.
    """


class _DescriptorFunctionRequired(DescriptorWindowUnsupported):
    """Exact unmet storage-only demand; the consumer cannot issue its token."""

    def __init__(self, procedure, expression, *, legacy_eligible=False):
        super().__init__("descriptor window requires registered original readonly source-function proof")
        self.procedure, self.expression = procedure, expression
        self.legacy_eligible = legacy_eligible


class _DescriptorFunctionsRequired(CompilationError):
    """Private discovery result; no incomplete window may be published."""

    def __init__(self, demands):
        super().__init__("descriptor window has unmet source-function requirements")
        self.demands = tuple(demands)


class _DescriptorTerminationRequired(DescriptorWindowUnsupported):
    """Exact terminating-call demand; a consumer cannot issue its own token."""

    def __init__(self, procedure, call, reason):
        super().__init__(reason)
        self.procedure, self.call = procedure, call


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


_ABSENT = object()


def _class_field(kind, name):
    for base in type.__getattribute__(kind, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return _ABSENT


def _raw_dictionary(value):
    descriptor = _class_field(type(value), "__dict__")
    if descriptor is _ABSENT:
        return None
    if type(descriptor) not in {GetSetDescriptorType, MemberDescriptorType}:
        raise CompilationError("descriptor window rejects delegated instance dictionary access")
    dictionary = type(descriptor).__get__(descriptor, value, type(value))
    if type(dictionary) is not dict:
        raise CompilationError("descriptor window requires a builtin instance dictionary")
    return dictionary


def _raw_value(value, name):
    dictionary = _raw_dictionary(value)
    if dictionary is not None and name in dictionary:
        return dictionary[name]
    descriptor = _class_field(type(value), name)
    if type(descriptor) in {GetSetDescriptorType, MemberDescriptorType}:
        return type(descriptor).__get__(descriptor, value, type(value))
    if descriptor is not _ABSENT and _class_field(type(descriptor), "__get__") is not _ABSENT:
        raise CompilationError("descriptor window rejects delegated field access")
    return descriptor


def _raw_fields(value, names=None):
    """Use builtin dictionaries/slots; never an instance getter or property."""
    kind, dictionary = type(value), _raw_dictionary(value)
    keys = tuple(dictionary) if dictionary is not None else None
    if names is None:
        names = list(keys or ())
        for base in type.__getattribute__(kind, "__mro__"):
            names.extend(name for name, item in type.__getattribute__(base, "__dict__").items()
                         if type(item) is MemberDescriptorType and name not in names)
        names = tuple(names)
    attached = tuple((name, _raw_value(value, name), _class_field(kind, name)) for name in names)
    return (value, kind, dictionary, keys if names == keys else None,
            _class_field(kind, "__getattribute__"), attached)


def _same_raw_fields(saved):
    value, kind, dictionary, keys, getter, attached = saved
    return (type(value) is kind and _class_field(kind, "__getattribute__") is getter
            and all(_class_field(kind, name) is descriptor for name, _item, descriptor in attached)
            and _raw_dictionary(value) is dictionary
            and (keys is None or tuple(dictionary) == keys)
            and all(_raw_value(value, name) is item for name, item, _descriptor in attached))


def _mapping_items(value):
    from compiler.frontend.immutable_graph import FrozenDict
    from compiler.frontend.source_session import ProjectedRoutines

    if type(value) in {dict, FrozenDict}:
        return tuple(dict.items(value))
    if type(value) is MappingProxyType:
        return tuple(value.items())
    if type(value) is ProjectedRoutines:
        original = dict(_mapping_items(_raw_value(value, "base")))
        dict.update(original, _mapping_items(_raw_value(value, "_overrides")))
        return tuple(dict.items(original))
    raise CompilationError("descriptor window requires trusted consuming mappings")


def _proof_registry_items(registry, *, terminating=False):
    """Authenticate a consumed digest registry before any hashing lookup."""
    if type(registry) is not dict:
        raise CompilationError("descriptor window requires its builtin source proof registry")
    items = tuple(dict.items(registry))
    if terminating:
        if any(type(key) is not tuple or len(key) != 2 or type(key[0]) is not str or type(key[1]) is not int
               for key, _proof in items):
            raise CompilationError("descriptor window requires exact terminating callsite registry keys")
    elif any(type(identity) is not str for identity, _proof in items):
        raise CompilationError("descriptor window requires exact string source proof registry keys")
    return items


def _sequence_items(value):
    if isinstance(value, tuple):
        return tuple(tuple.__iter__(value))
    if isinstance(value, list):
        return tuple(list.__iter__(value))
    if type(value) is set:
        return tuple(set.__iter__(value))
    return tuple(frozenset.__iter__(value))



def _same_container(value, runtime_type, kind, items):
    if type(value) is not runtime_type:
        return False
    current = _mapping_items(value) if kind == "mapping" else _sequence_items(value)
    if len(current) != len(items):
        return False
    if kind == "mapping":
        return all(key is saved_key and item is saved_item
                   for (key, item), (saved_key, saved_item) in zip(current, items, strict=True))
    if kind == "set":
        return all(any(item is now for now in current) for item in items)
    return all(left is right for left, right in zip(current, items, strict=True))


class _PublicationAttachments:
    """One derivation's raw consuming attachments, never validation authority.

    No lookup, public serializer, source formatter or proof callback runs while
    taking or comparing this receipt. Only reached namespaces and consumed
    tokens are expanded; the analysis and its caches remain opaque objects.
    """

    def __init__(self, analysis, check=None, query=None):
        from compiler.frontend.immutable_graph import FrozenDict, FrozenList
        from compiler.frontend.lowering import _KindScope
        from compiler.frontend.sealed_syntax import FrozenList as SyntaxFrozenList
        from compiler.frontend.source_facts import SourceFact
        from compiler.frontend.source_session import _ROOTS, ProjectedRoutines, SourceSession

        self.analysis = analysis
        self.root_policy = _ROOTS
        trusted_maps = {dict, MappingProxyType, FrozenDict, ProjectedRoutines}
        trusted_sequences = {tuple, list, set, frozenset, FrozenList, SyntaxFrozenList}
        # The final pass must not dispatch to foreign mapping/sequence hooks.
        # Authenticate root container types before even taking their entries.
        for name in _ROOTS:
            value = _raw_value(analysis, name)
            if isinstance(value, Mapping) and type(value) not in trusted_maps:
                raise CompilationError("descriptor window requires original trusted source containers")
            if type(value) is ProjectedRoutines and (
                    type(_raw_value(value, "base")) not in {dict, MappingProxyType, FrozenDict}
                    or type(_raw_value(value, "_overrides")) not in {dict, MappingProxyType, FrozenDict}):
                raise CompilationError("descriptor window requires original projected source tables")
        session = _raw_value(analysis, "_source_session")
        self.session = session
        self.source = (None if session is _ABSENT else session, tuple(
            (name, value, _mapping_items(value) if isinstance(value, Mapping) else
             _sequence_items(value) if type(value) in trusted_sequences else None)
            for name in _ROOTS for value in (_raw_value(analysis, name),)))
        self.fields, self.containers, self.registrations = [], [], []
        self.inputs = _raw_value(analysis, "inputs")
        input_fields = tuple((name, _raw_value(self.inputs, name)) for name in
                             ("paths", "sources", "_texts", "entries", "dependencies", "configuration"))
        self.fields.append(_raw_fields(self.inputs, tuple(name for name, _value in input_fields)))
        policy = tuple((name, _raw_value(analysis, name)) for name in
                       ("depth_limit", "procedure_limit", "operation_limit", "contracts", "_allocation_authorizations",
                        "_stable_module_allocatables", "_native_metadata", "_selected_descriptor_window"))
        self.fields.append(_raw_fields(analysis, tuple(name for name, _value in policy)))
        # Fields below are retained source and proof attachments, not the
        # charged scanner counters or mutable proof-builder caches.
        retained = ("scope_roles", "checked", "functions", "descriptor_functions", "storage_functions",
                    "intrinsics", "intrinsic_tokens", "predicates", "prefixes", "joins", "environments",
                    "environment_states", "objects", "object_mappings", "metadata_objects", "metadata_calls",
                    "metadata_schemas", "fixed_module_scopes", "fixed_module_bindings", "scalar_clocks",
                    "scalar_clock_tokens", "storage_groups", "storage_group_tokens", "library_states",
                    "library_calls", "fixed_associations", "numeric_component_reads", "joined_blocks",
                    "private_allocations", "expression_tokens", "storage_teams", "record_calls", "record_numeric_allocations")
        self.check = check
        self.check_fields = tuple((name, _raw_value(check, name)) for name in retained) if check is not None else ()
        pending = [*(dict.values(_raw_value(check, "resolution_scopes")) if check is not None else (query,)),
                   *(value for _name, value in input_fields),
                   *(value for name, value in self.check_fields if name not in {"private_allocations", "joined_blocks", "expression_tokens"}),
                   *(value for _name, value in policy),
                   self.source[0], *(value for _name, value, _contents in self.source[1])]
        # Existing typed checkpoint baselines belong to their own validators.
        # Retain their exact outer receipt without recursively expanding their
        # whole-source audit snapshots or opaque parser-reader caches.
        for mapping in ((check.private_allocations, check.joined_blocks) if check is not None else ()):
            self.containers.append((mapping, type(mapping), "mapping", _mapping_items(mapping)))
            pending.extend(proof for _key, (proof, _baseline) in _mapping_items(mapping))
        self.expressions = tuple(check.expression_tokens.items()) if check is not None else ()
        if check is not None:
            mapping = check.expression_tokens
            self.containers.append((mapping, type(mapping), "mapping", self.expressions))
            pending.extend(entry[4] for _key, entry in self.expressions)
            for _key, entry in self.expressions:
                registry = entry[2]
                self.containers.append((registry, dict, "mapping", _proof_registry_items(registry)))
                # A storage expression also consumes its exact ordinary and
                # terminating call registries. Nested token/map fields alone
                # do not retain the current owner pointer after callbacks.
                calls = _raw_value(entry[4], "_call_registries")
                if calls is not _ABSENT:
                    if (type(calls) is not tuple or any(type(row) is not tuple or len(row) != 2
                            or type(row[0]) is not str or type(row[1]) is not dict for row in calls)
                            or any(_raw_value(entry[0], name) is not original for name, original in calls)):
                        raise CompilationError("descriptor expression consumed call registry attachment changed")
                    self.fields.append(_raw_fields(entry[0], tuple(name for name, _registry in calls)))
        role_maps = [value for name, value in self.check_fields if name == "scope_roles" and isinstance(value, Mapping)]
        seen, registered, registry_views = set(), set(), {}
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, Mapping):
                if type(value) not in trusted_maps:
                    raise CompilationError("descriptor window requires trusted consuming mappings")
                if type(value) is ProjectedRoutines:
                    self.fields.append(_raw_fields(value, ("base", "_overrides")))
                    if type(_raw_value(value, "base")) not in trusted_maps or type(_raw_value(value, "_overrides")) not in {dict, MappingProxyType, FrozenDict}:
                        raise CompilationError("descriptor window requires original projected source tables")
                items = _mapping_items(value)
                self.containers.append((value, type(value), "mapping", items))
                # Root tables retain exact entries without expanding every
                # unrelated procedure. Reached scopes below own the deep walk.
                if (any(value is root for _name, root, _contents in self.source[1])
                        or any(value is role_map for role_map in role_maps)):
                    continue
                pending.extend(item for _key, item in items)
            elif isinstance(value, (tuple, list, set, frozenset)):
                if type(value) not in trusted_sequences:
                    raise CompilationError("descriptor window requires trusted consuming sequences")
                items = _sequence_items(value)
                self.containers.append((value, type(value), "set" if isinstance(value, (set, frozenset)) else "sequence", items))
                # Original root inventories (notably kind_expressions) retain
                # exact entries without walking every unrelated declaration
                # and its owner. Consulted scope/binding/kind roots below are
                # still expanded and checked independently.
                if any(value is root for _name, root, _contents in self.source[1]):
                    continue
                pending.extend(items)
            elif isinstance(value, Base):
                state = _raw_fields(value)
                self.fields.append(state)
                attached = tuple((name, item) for name, item, _descriptor in state[-1])
                # Parent identities are retained without walking back out of
                # the reached source tree. Parser item metadata is raw too.
                if type(value).__name__ in {"Module", "Submodule"}:
                    # A consulted parent module contributes declarations and
                    # exact member identities, not every sibling procedure's
                    # body. Reached routine scopes/graphs retain their own
                    # complete source independently below.
                    pending.extend(item for name, item in attached if name not in {"parent", "item", "content"})
                    content = _raw_value(value, "content")
                    if type(content) not in trusted_sequences:
                        raise CompilationError("descriptor window requires original module member containers")
                    children = _sequence_items(content)
                    self.containers.append((content, type(content), "sequence", children))
                    for child in children:
                        if type(child).__name__ == "Module_Subprogram_Part":
                            member_state = _raw_fields(child)
                            self.fields.append(member_state)
                            members = _raw_value(child, "content")
                            if type(members) not in trusted_sequences:
                                raise CompilationError("descriptor window requires original module member containers")
                            entries = _sequence_items(members)
                            self.containers.append((members, type(members), "sequence", entries))
                            pending.extend(node for node in entries if type(node).__name__ == "Contains_Stmt")
                        else:
                            pending.append(child)
                else:
                    pending.extend(item for name, item in attached if name not in {"parent", "item"})
                item = _raw_value(value, "item")
                if item is not _ABSENT and item is not None:
                    self.fields.append(_raw_fields(item))
                    # Reader replacement caches are not original syntax.
                    # Retain their pointer, without invoking/expanding their
                    # custom container implementations at publication.
            elif not isinstance(value, type) and _class_field(type(value), "__dataclass_fields__") is not _ABSENT:
                state = _raw_fields(value)
                self.fields.append(state)
                attached = tuple((name, item) for name, item, _descriptor in state[-1])
                # Session objects are immutable authority anchors. Expanding
                # their complete object inventory would walk unrelated source.
                if type(value) not in {SourceSession, SourceFact}:
                    for name, item in attached:
                        if name == "_roles" and isinstance(item, Mapping):
                            # A role inventory retains exact entries; only the
                            # independently consulted scopes and proof source
                            # objects below are expanded into source receipts.
                            role_maps.append(item)
                        pending.append(item)
                owner = _raw_value(value, "_analysis")
                # Immutable arena dependencies are not analysis-registered
                # proofs. Do not inspect an unrelated identity property.
                identity = _raw_value(value, "identity") if owner is not _ABSENT and owner is not None else _ABSENT
                if owner is not _ABSENT and owner is not None and type(identity) is str:
                    for name, registry in _raw_dictionary(owner).items():
                        if type(name) is not str or not name.startswith("_") or type(registry) is not dict:
                            continue
                        # Discover actual ownership by raw value identity. A
                        # lookup in an unrelated map could dispatch a foreign
                        # colliding key before we even identify the registry.
                        if id(registry) not in registry_views:
                            registry_views[id(registry)] = tuple(dict.items(registry))
                        candidates = registry_views[id(registry)]
                        if not any(proof is value for _key, proof in candidates):
                            continue
                        terminating = name == "_descriptor_terminating_calls"
                        items = _proof_registry_items(registry, terminating=terminating)
                        key = ((_raw_value(value, "_procedure"), id(_raw_value(value, "_call")))
                               if terminating else identity)
                        if (any(entry == key and proof is value for entry, proof in items)
                                and (id(owner), name, key) not in registered):
                            registered.add((id(owner), name, key))
                            self.containers.append((registry, dict, "mapping", items))
                            self.registrations.append((owner, name, registry, key, value,
                                                       _raw_value(owner, name + "_owner"), terminating))
            elif isinstance(value, _KindScope):
                state = _raw_fields(value, ("parent", "values", "intrinsic_kinds"))
                self.fields.append(state)
                attached = tuple((name, item) for name, item, _descriptor in state[-1])
                pending.extend(item for _name, item in attached)

    def require(self):
        from compiler.frontend import source_session

        if (source_session._ROOTS is not self.root_policy or _raw_value(self.analysis, "inputs") is not self.inputs
                or any(not _same_raw_fields(state) for state in self.fields)
                or any(_raw_value(self.check, name) is not value for name, value in self.check_fields)
                or _raw_value(self.analysis, "_source_session") is not self.session
                or any(_raw_value(self.analysis, name) is not value
                       for name, value, _contents in self.source[1])
                or any(not _same_container(value, runtime_type, kind, items)
                       for value, runtime_type, kind, items in self.containers)
                or any(_raw_value(owner, name) is not registry
                       or not _proof_registry_items(registry, terminating=terminating) or dict.get(registry, identity) is not proof
                       or _raw_value(owner, name + "_owner") is not marker
                       for owner, name, registry, identity, proof, marker, terminating in self.registrations)):
            raise CompilationError("descriptor window consuming source attachment changed at publication")
        for _key, (owner, name, registry, identity, proof, marker, receipt) in self.expressions:
            if (_raw_value(owner, name) is not registry or not _proof_registry_items(registry)
                    or dict.get(registry, identity) is not proof or _raw_value(owner, name + "_owner") is not marker):
                raise CompilationError("descriptor expression source token registration changed at publication")
            receipt.require()


class _FactoryAttachments:
    """Baseline registered tokens before the first callbackful derivation."""

    def __init__(self, analysis, procedure):
        self.analysis = analysis
        registry = _raw_value(analysis, "_descriptor_window_proofs")
        if registry is _ABSENT:
            registry = None
        if registry is None:
            registry = analysis._descriptor_window_proofs = {}
        if type(registry) is not dict:
            raise CompilationError("descriptor window requires its compiler-owned proof registry")
        self.registry = registry
        self.entries = _proof_registry_items(registry)
        self.tokens = tuple(_raw_fields(proof) for _identity, proof in self.entries
                            if type(proof) is DescriptorWindowProof)
        self.owner = _raw_value(analysis, "_descriptor_window_proofs_owner")
        from compiler.frontend.immutable_graph import FrozenDict
        from compiler.frontend.source_session import ProjectedRoutines

        routines = _raw_value(analysis, "routines")
        if type(routines) not in {dict, MappingProxyType, FrozenDict, ProjectedRoutines}:
            raise CompilationError("descriptor window requires its original source routine table")
        if type(procedure) is not str:
            raise CompilationError("descriptor window requires its original source procedure")
        if any(type(name) is not str for name, _routine in _mapping_items(routines)):
            raise CompilationError("descriptor window requires original string procedure names")
        routine = next((routine for name, routine in _mapping_items(routines) if name == procedure), None)
        self.query = _PublicationAttachments(analysis, query=routine)

    def require(self):
        registry = self.registry
        current = _proof_registry_items(registry)
        self.query.require()
        if (_raw_value(self.analysis, "_descriptor_window_proofs") is not registry
                or _raw_value(self.analysis, "_descriptor_window_proofs_owner") is not self.owner
                or len(current) != len(self.entries)
                or any(key is not identity or value is not proof
                       for (key, value), (identity, proof) in zip(current, self.entries, strict=True))
                or any(not _same_raw_fields(state) for state in self.tokens)):
            raise CompilationError("descriptor window registered interval changed during fresh derivation")


def _publish_derivation(analysis, procedure, authority, record, check):
    check.publish(procedure, authority)
    # Preserve the historical counter endpoints: publication validations are
    # charged, whereas preparing public payloads is not a source traversal.
    record["source_operations"] = check.operations
    record["source_traversal_entries"] = check.nodes
    record["resolution_validation"].update(units=check.resolution_units, work_entries=check.resolution_work,
        syntax_entries=check.resolution_syntax, lookup_transitions=check.resolution_lookups)


def _finish_derivation(analysis, attachments):
    # Even a last role/serializer callback can change a live file. Nothing
    # callbackful follows this verification: the remaining fence is raw only.
    analysis.inputs.verify_live()
    attachments.require()


class _MetadataTypes(_TypeBuilder):
    """Retain the exact scopes consulted by one metadata type resolution."""
    def __init__(self, analysis):
        super().__init__(analysis)
        self.recording = None

    def scope(self, scope):
        record = super().scope(scope)
        if self.recording is not None:
            self.recording[id(scope)] = (scope, record)
        return record


@dataclass(frozen=True)
class _MetadataSchemaReceipt:
    check: object
    analysis: object
    binding: object
    declaring_scope: object
    type_name: str
    schema: object
    scopes: tuple
    limits: tuple
    session: object


class _Check:
    def __init__(self, analysis):
        self.analysis = analysis
        self.scope_roles = _scope_roles(analysis)
        self.source_budget = DescriptorSourceBudget(analysis)
        self.nodes = self.operations = 0
        # operations is an exact descriptive count. Full original syntax and
        # per-semantic-unit resolution retain their independent hard bounds.
        self.resolution_units = self.resolution_work = 0
        self.resolution_syntax = self.resolution_lookups = 0
        self.resolution_scopes = {}
        self.active, self.checked = [], {}
        self.functions = {}
        self.expression_tokens = {}
        self.function_demands = None
        self.descriptor_functions = {}
        self.storage_functions = {}
        self.intrinsics = {}
        self.intrinsic_tokens = {}
        self.intrinsic_authority = None
        self.predicates = {}
        self.prefix_scopes, self.prefixes = {}, {}
        self.joined_directives, self.joins = {}, {}
        self.joined_returns, self.native_returns = set(), {}
        self.joined_loop_controls, self.loop_frames = set(), []
        self.native_loops, self.native_loop_controls = {}, {}
        self.storage_teams = []
        self.record_calls = {}
        self.record_numeric_allocations = ()
        self.team_calls, self.team_call_depth = set(), 0
        # Register source-derived group tokens only in this private view. Its
        # original procedures, bindings and parser nodes retain their exact
        # authority; no default summary or caller-owned proof registry changes.
        from compiler.scopes.segments import _native_analysis

        self.join_analysis = _native_analysis(analysis)
        # Completion identities include the analysis policy. An unrelated
        # previously issued team token must not be consumed by this selected
        # descriptor query after native policy changes. Discover selected
        # groups again in this private issuer view under its current authority.
        self.join_analysis._joined_completions = {}
        self.join_analysis._numerical_completions = {}
        self.join_analysis._owned_record_regions = {}
        self.join_analysis._owned_record_regions_owner = self.join_analysis
        # Proof builders must register in this view, not follow a projection
        # backlink and mutate its original analysis. Every accepted procedure
        # still passes _require_original against the retained source roles.
        self.join_analysis._descriptor_source = None
        self.environments, self.environment_states = {}, {}
        self.object_builder = _MetadataTypes(analysis)
        self.objects, self.object_schemas, self.object_mappings = {}, {}, {}
        self.metadata_objects = {}
        self.metadata_calls = {}
        self.metadata_schemas = {}
        self.optional_scalars = {}
        self.optional_arrays = {}
        self.terminating_calls = {}
        self.native_outputs = {}
        self.fixed_module_scopes = {}
        self.fixed_module_bindings = {}
        self.scalar_clocks, self.scalar_clock_tokens = {}, {}
        self.storage_groups, self.storage_group_tokens = {}, {}
        self.worksharing_storage, self.worksharing_directives = {}, {}
        self.library_states, self.library_calls = {}, {}
        self.fixed_associations = {}
        self.association_stack = []
        self.numeric_component_reads = {}
        self.joined_blocks = {}
        self.protected_allocations = frozenset()
        self.private_allocations = {}
        self.private_allocation_source = None

    def retain_expression(self, proof, owner, registry_name, routine, procedures=()):
        """Retain one consumed expression token before its validation callbacks."""
        identity, registry = _raw_value(proof, "identity"), _raw_value(owner, registry_name)
        if type(identity) is not str or not _proof_registry_items(registry) or dict.get(registry, identity) is not proof:
            raise CompilationError("descriptor expression requires its exact registered source token")
        key = registry_name, identity
        if key not in self.expression_tokens:
            roots = [routine]
            helpers, routines = _raw_value(owner, "numerical_helpers"), _raw_value(owner, "routines")
            for procedure in procedures:
                original = next((value for name, value in (*_mapping_items(helpers), *_mapping_items(routines))
                                 if name == procedure), None)
                if original is None:
                    raise CompilationError("descriptor expression requires its complete original source closure")
                roots.append(original)
            receipt = _PublicationAttachments(owner, query=(proof, *roots))
            self.expression_tokens[key] = (owner, registry_name, registry, identity, proof,
                                           _raw_value(owner, registry_name + "_owner"), receipt)
        owner0, name0, registry0, identity0, proof0, marker, receipt = self.expression_tokens[key]
        _proof_registry_items(registry)
        if (owner is not owner0 or registry_name != name0 or registry is not registry0 or proof is not proof0
                or identity != identity0 or _raw_value(owner, registry_name + "_owner") is not marker):
            raise CompilationError("descriptor expression source token attachment changed")
        receipt.require()

    def private_allocation_mutation(self, routine, binding):
        """Preserve other roots only under a final unregistered-storage duty."""
        if (binding.declaring_scope is not routine.scope or binding.name in routine.arguments
                or not {"save", "allocatable"} <= binding.attributes
                or binding.root in self.protected_allocations):
            return False
        from compiler.frontend.local_allocatable_storage import allocation_source_checkpoint

        if self.private_allocation_source is None:
            self.private_allocation_source = allocation_source_checkpoint(self.analysis)
        key = routine.qualified, binding.root
        if key not in self.private_allocations:
            teams = {}
            for (procedure, _node), joined in self.joined_directives.items():
                if procedure != routine.qualified or id(joined) in teams:
                    continue
                if type(joined) is DescriptorTeamProof:
                    proof = joined
                else:
                    from compiler.frontend.descriptor_teams import _original_group

                    graph = self.analysis.structure(routine.qualified)
                    originals = tuple(node for identity in joined.selected_node_ids
                                      for node in graph.source_nodes(identity))
                    group = _original_group(routine, originals, graph_selection=True)
                    proof = prove_descriptor_team(self.join_analysis, routine.qualified, group.original_nodes)
                teams[id(joined)] = proof
            proof = original_local_allocation(self.join_analysis, routine, binding,
                                              joined_storage=tuple(teams.values()))
            self.private_allocations[key] = proof, proof.checkpoint()
        return True

    def native_intrinsic(self, routine, node, classifier, *, names=()):
        """Retain declaration authority before an intrinsic consumes types."""
        from compiler.frontend.descriptor_associations import _attachments, _syntax_attachments
        from compiler.frontend.descriptor_intrinsics import intrinsic_scope_attachments, intrinsic_source_attachments
        from compiler.frontend.source_functions import _resolution

        syntax = _syntax_attachments(routine.scope.node)
        routine_fields = _attachments(routine)
        if self.intrinsic_authority is None:
            self.intrinsic_authority = intrinsic_source_attachments(self.analysis)
        scopes = {}

        def retain_scope(scope):
            if id(scope) not in scopes:
                scopes[id(scope)] = intrinsic_scope_attachments(scope)
                authority = self.analysis._source_function_scopes.get(id(scope))
                if (authority is None or authority[0] is not scope or authority[1] is not scope.node
                        or authority[2] is not scope.parent or authority[3] != _resolution(scope)):
                    raise CompilationError("descriptor intrinsic requires unchanged original namespace")
            self.scope_authority(scope)

        # Include the intrinsic's own lookup even if it consumes no binding,
        # plus exact imported/host/lexical namespaces consulted by all actuals.
        scope = self.analysis.source_scope_for(node, routine.scope) if names else routine.scope
        _validate_resolution(self.analysis, scope, _walk(node), names=names, _scope_validator=retain_scope)
        bindings, pending = [], [node]
        while pending:
            original = pending.pop()
            if _kind(original) == "Name":
                scope = self.analysis.source_scope_for(original, routine.scope)
                binding = self.analysis._binding(scope, original)
                if binding is not None:
                    attached = _attachments(binding)
                    declaration = binding.declaring_scope
                    # A sealed namespace authenticates object identity, not a
                    # Binding whose dataclass fields were subsequently changed.
                    retain_scope(declaration)
                    authority = self.analysis._source_function_scopes.get(id(declaration))
                    record = None if authority is None else authority[4].get(binding.name)
                    if (authority is None or authority[0] is not declaration
                            or authority[1] is not declaration.node or authority[2] is not declaration.parent
                            or record is None or record[0] is not binding or record[1] is not declaration):
                        raise CompilationError("descriptor intrinsic requires unchanged original operand declaration")
                    bindings.append((original, scope, scope.bindings, binding, attached,
                                     declaration, declaration.bindings, _canonical(binding.public()),
                                     _syntax_attachments(binding.shape_nodes)))
            elif _kind(original) in {"Actual_Arg_Spec", "Intrinsic_Function_Reference"}:
                pending.append(original.items[1])
            elif isinstance(original, (Base, tuple, list)):
                pending.extend(reversed(_children(original)))
        actuals, record = classifier(self.analysis, routine, node)
        key = routine.qualified, id(node)
        receipt = (routine, node, classifier, _canonical(record), tuple(actuals), syntax,
                   routine_fields, tuple(bindings), tuple(scopes.values()))
        self.intrinsics[key] = record
        self.intrinsic_tokens[key] = receipt
        self.validate_native_intrinsics(((key, receipt),), rederive=False)
        return actuals

    def validate_native_intrinsics(self, retained=None, *, rederive=True, authority=None):
        """Validate consumed inputs and outputs, including the last callback."""
        from compiler.frontend.descriptor_associations import _attached, _same_syntax
        from compiler.frontend.descriptor_intrinsics import (
            same_intrinsic_scope_attachments,
            same_intrinsic_source_attachments,
        )

        retained = tuple(self.intrinsic_tokens.items()) if retained is None else retained
        authority = self.intrinsic_authority if authority is None else authority
        if rederive:
            for _key, (routine, node, classifier, public, actuals, _syntax, _fields, bindings, _scopes) in retained:
                for original, scope, _map, binding, _fields, declaration, _declarations, record, _axes in bindings:
                    if (self.analysis._binding(scope, original) is not binding
                            or binding.declaring_scope is not declaration
                            or _canonical(binding.public()) != record):
                        raise CompilationError("descriptor intrinsic original operand resolution changed")
                current, record = classifier(self.analysis, routine, node)
                if (_canonical(record) != public or len(current) != len(actuals)
                        or any(left is not right for left, right in zip(current, actuals, strict=True))):
                    raise CompilationError("descriptor intrinsic typed native effects changed at publication")
        # No resolution, signature or other charged callback follows this
        # cheap pass. Retained references prevent equivalent replacements and
        # earlier-object mutation by the final callback from reseeding a fence.
        for key, receipt in retained:
            routine, _node, _classifier, public, _actuals, syntax, attached, bindings, scopes = receipt
            if (self.intrinsic_tokens.get(key) is not receipt or _canonical(self.intrinsics.get(key)) != public
                    or not _attached(routine, attached) or not _same_syntax(syntax)):
                raise CompilationError("descriptor intrinsic original source attachment changed at publication")
            if any(not same_intrinsic_scope_attachments(state) for state in scopes):
                raise CompilationError("descriptor intrinsic original namespace attachment changed at publication")
            for _original, scope, namespace, binding, binding_fields, declaration, declarations, record, axes in bindings:
                if (scope.bindings is not namespace or declaration.bindings is not declarations
                        or declarations.get(binding.name) is not binding
                        or not _attached(binding, binding_fields) or binding.declaring_scope is not declaration
                        or _canonical(binding.public()) != record or not _same_syntax(axes)):
                    raise CompilationError("descriptor intrinsic operand declaration attachment changed at publication")
        if authority is not None and (self.intrinsic_authority is not authority
                or not same_intrinsic_source_attachments(self.analysis, authority)):
            raise CompilationError("descriptor intrinsic original source registry attachment changed at publication")

    def numeric_component_read(self, routine, node):
        """Inspect an unchanged native read without requiring payload layout."""
        from compiler.frontend.descriptor_associations import _same_syntax, _syntax_attachments
        from compiler.frontend.storage_component_reads import numeric_component_read

        scopes = {}
        original_syntax, original_text = _syntax_attachments(node), str(node)

        def scope_authority(scope):
            self.scope_authority(scope)
            scopes[id(scope)] = scope

        component = numeric_component_read(self.analysis, routine, node,
            visit_index=lambda index: self.expression(routine, index),
            visit_scope=scope_authority, tick=self.tick)
        if not _same_syntax(original_syntax) or str(node) != original_text:
            raise CompilationError("descriptor numeric component read changed during original traversal")
        self.numeric_component_reads[routine.qualified, id(node)] = (
            routine, node, component.root_binding, component.indices,
            component._record, tuple(scopes.values()), original_syntax)

    def validate_numeric_component_reads(self):
        from compiler.frontend.descriptor_associations import _same_syntax
        from compiler.frontend.storage_component_reads import numeric_component_read

        retained = tuple(self.numeric_component_reads.items())
        for _key, (routine, node, root, indices, record, original_scopes, syntax) in retained:
            if not _same_syntax(syntax):
                raise CompilationError("descriptor numeric component original parser attachments changed")
            scopes = {}
            # The ordinary scanner already charged and checked every index.
            # Revalidate the same original declaration/read occurrence after
            # all charged callbacks, without recursively granting new work.
            component = numeric_component_read(self.analysis, routine, node,
                visit_index=lambda _index: None,
                visit_scope=lambda scope, scopes=scopes: scopes.setdefault(id(scope), scope), tick=lambda: None)
            current_scopes = tuple(scopes.values())
            if (not _same_syntax(syntax) or component.root_binding is not root or component._record != record
                    or len(component.indices) != len(indices)
                    or any(left is not right for left, right in zip(component.indices, indices, strict=True))
                    or len(current_scopes) != len(original_scopes)
                    or any(left is not right for left, right in zip(current_scopes, original_scopes, strict=True))):
                raise CompilationError("descriptor numeric component read changed at source publication")
        if (len(self.numeric_component_reads) != len(retained)
                or any(self.numeric_component_reads.get(key) is not receipt for key, receipt in retained)):
            raise CompilationError("descriptor numeric component read attachment changed at source publication")

    def library_constructor(self, routine, assignment):
        from compiler.frontend.call_bindings import resolve_original_function

        for proof in self.library_states.values():
            declaration = proof._declaration
            if declaration._routine is routine and declaration._constructor._assignment is assignment:
                proof.validate()
                expression = assignment.items[2]
                resolved = resolve_original_function(self.analysis, routine.scope, expression)
                self.resolution(routine, expression)
                for actual in resolved.actuals:
                    self.expression(routine, actual)
                return True
        return False

    def library_call(self, routine, graph, identity, call):
        from compiler.frontend.library_storage_calls import library_storage_call
        from compiler.frontend.storage_mutation_effects import _preserves_record, prove_storage_mutation_effects

        proof = library_storage_call(self.analysis, routine, call)
        if proof is None:
            return False
        storage = prove_storage_mutation_effects(self.analysis, routine.qualified, selection=(identity,))
        if not _preserves_record(storage._record):
            raise DescriptorWindowUnsupported("qualified library storage CALL has unproved mutation or escape")
        arrays = {item["name"] for item in proof.public()["array_formals"]}
        for item in proof._resolved.associations:
            if item.formal in arrays:
                self.expression(routine, item.actual)
        self.library_calls[(routine.qualified, identity)] = (proof, storage)
        return True

    def storage_group(self, routine, graph, identity):
        from compiler.frontend.descriptor_storage_groups import original_storage_group

        group = graph.native_groups.get(identity)
        if group is None:
            return False
        originals = graph.source_nodes(identity)
        if not group.matches(originals):
            raise DescriptorWindowUnsupported("descriptor group lost its complete original source selection")
        private_graph = self.join_analysis.structure(routine.qualified)
        private_group = private_graph.native_group_for_selection(originals)
        if private_group is None or private_group.node_id != identity:
            raise DescriptorWindowUnsupported("descriptor group changed its exact private source membership")
        proof = original_storage_group(self.join_analysis, routine, private_graph, identity)
        key = routine.qualified, identity
        self.storage_group_tokens[key] = (routine, graph, group, proof)
        self.storage_groups[key] = {"procedure": routine.qualified, "source_graph": graph.identity,
            "original_node_id": identity, "storage_group": proof.public()}
        return True

    def scalar_clock(self, routine, call):
        """Existing host-clock authority establishes scalar-only storage effects."""
        from compiler.frontend.scalar_clock import is_system_clock, original_scalar_clock

        if _kind(call) != "Call_Stmt":
            return False
        try:
            standard = is_system_clock(self.analysis, self.analysis.source_scope_for(call, routine.scope), call)
        except CompilationError:
            # A same-named source procedure retains its ordinary full-closure
            # path. Shadowing never grants standard-intrinsic authority.
            return False
        if not standard:
            return False
        self.resolution(routine, call)
        proof = original_scalar_clock(self.join_analysis, routine.qualified, call)
        public = proof.public()
        if _hash({key: value for key, value in public.items() if key != "identity"}) != proof.identity:
            raise CompilationError("descriptor scalar clock changed its source content identity")
        self.scalar_clocks[proof.identity] = {**public, "descriptor_evidence_only": True,
                                            "execution_authorized": False}
        self.scalar_clock_tokens[id(call)] = (routine, call, proof)
        return True

    def joined_block(self, routine, graph, identity, original):
        from compiler.frontend.descriptor_block_storage import _team_syntax, original_joined_block_storage

        # The descriptor-only team is independent of numerical group size.
        # Declaration scope alone cannot discharge an ordinary BLOCK boundary.
        candidates = {id(proof): proof for proof in self.joined_directives.values()
                      if type(proof) is DescriptorTeamProof and proof.procedure == routine.qualified
                      and any(node is original for node, _children in _team_syntax(proof._originals))}
        if len(candidates) != 1:
            raise DescriptorWindowUnsupported("descriptor BLOCK requires one exact complete original joined storage team")
        team, = candidates.values()
        private_graph = self.join_analysis.structure(routine.qualified)
        actual = graph.source_nodes(identity)
        private = private_graph.source_nodes(identity) if identity in private_graph.nodes else ()
        if len(actual) != 1 or actual[0] is not original or len(private) != 1 or private[0] is not original:
            raise CompilationError("descriptor BLOCK changed its original graph membership")
        proof = original_joined_block_storage(self.join_analysis, routine, private_graph, identity, original, team,
            scope_validator=self.scope_authority, tick=self.tick)
        self.joined_blocks[routine.qualified, identity] = (proof, proof.checkpoint())
        return True

    @staticmethod
    def optional_scalar(routine, binding):
        # Reuse the original-body companion's exact scalar signature policy.
        # This descriptor scan never evaluates/snapshots an optional payload.
        from compiler.scopes.lexical import optional_scalar_input

        return (binding is not None and binding.name in routine.arguments
                and routine.scope.bindings.get(binding.name) is binding
                and binding.declaring_scope is routine.scope and "target" not in binding.attributes
                and optional_scalar_input(binding))

    @staticmethod
    def optional_output(routine, binding):
        from compiler.frontend.descriptor_optional import optional_numeric_output

        return optional_numeric_output(routine, binding)

    def optional_scalar_mapping(self, routine, call, resolved, mapping):
        child = self.analysis.routines[resolved.procedure]
        formal = mapping.formal_binding
        output = self.optional_output(child, formal)
        if not self.optional_scalar(child, formal) and not output:
            return None
        bounds = self.fixed_vector_bounds(child, formal) if output and formal.rank else None
        actual, binding = mapping.actual, mapping.binding
        if mapping.presence == "omitted":
            valid = actual is None and binding is None
        elif output:
            from compiler.frontend.descriptor_optional import independent_scalar_output

            # The first array slice is absence-only: it does not authorize an
            # optional vector capture, association, section or payload view.
            valid = not formal.rank and _kind(actual) == "Name" and (
                self.optional_output(routine, binding) and not binding.rank
                if mapping.presence == "forwarded_optional" else
                mapping.presence == "supplied" and independent_scalar_output(self, routine, binding))
            if binding is not None and any(other is not mapping and other.binding is binding for other in resolved.mappings):
                valid = False
        elif mapping.presence == "forwarded_optional":
            valid = _kind(actual) == "Name" and self.optional_scalar(routine, binding)
        elif mapping.presence == "supplied":
            valid = actual is not None and (binding is None or not binding.rank
                and not binding.attributes & (_UNSAFE | {"allocatable", "target"}))
        else:
            valid = False
        if not valid:
            return None
        record = {"caller": routine.qualified, "callee": resolved.procedure,
                  "original_call": str(call), "formal": mapping.formal, "presence": mapping.presence,
                  "formal_descriptor": mapping.formal_binding.public(),
                  "actual": str(actual) if actual is not None else None,
                  "actual_descriptor": binding.public() if binding is not None else None,
                  "original_presence_and_evaluation_required": True,
                  "value_evaluation_authorized": False, "payload_access_authorized": False,
                  "entry_definition_authority": False, "execution_authorized": False}
        if output:
            record["original_positive_present_required"] = True
        if bounds is not None:
            record["fixed_formal_bounds"] = list(bounds)
        (self.optional_arrays if bounds is not None else self.optional_scalars)[routine.qualified, id(call), mapping.formal] = record
        return record

    def terminating_call(self, routine, call):
        from compiler.frontend.descriptor_termination import DescriptorTerminatingCallProof

        proof = getattr(self.analysis, "_descriptor_terminating_calls", {}).get((routine.qualified, id(call)))
        if proof is None:
            return False
        if not isinstance(proof, DescriptorTerminatingCallProof):
            raise CompilationError("descriptor terminating call requires its exact original proof type")
        proof.validate(self.analysis, routine.qualified, call, tick=self.tick, resolution_tick=self.resolution_tick)
        self.terminating_calls[proof.identity] = proof.public()
        return True

    def metadata_schema(self, binding, type_name):
        receipt = self.metadata_schemas.get(id(binding))
        if receipt is None:
            scopes = {}
            self.object_builder.recording = scopes
            try:
                schema = self.object_builder.resolve(binding.declaring_scope, type_name)
            finally:
                self.object_builder.recording = None
            if schema is None or schema.reason:
                raise DescriptorWindowUnsupported("descriptor metadata requires its exact nonpolymorphic declaring type")
            receipt = _MetadataSchemaReceipt(self, self.analysis, binding, binding.declaring_scope, type_name, schema,
                tuple((scope, record, tuple((name, value[0], value[0].private, value[0].reason)
                      for name, value in record[5].items())) for scope, record in scopes.values()),
                (self.analysis.procedure_limit, self.analysis.depth_limit, self.analysis.operation_limit),
                getattr(self.analysis, "_source_session", None))
        if (receipt.check is not self or receipt.analysis is not self.analysis or receipt.binding is not binding
                or receipt.declaring_scope is not binding.declaring_scope or receipt.type_name != type_name
                or receipt.limits != (self.analysis.procedure_limit, self.analysis.depth_limit, self.analysis.operation_limit)
                or receipt.session is not getattr(self.analysis, "_source_session", None)):
            raise CompilationError("descriptor metadata type receipt lost its original source authority")
        # Authenticate scopes afresh; this helper only checks declarations and
        # never calls resolve or resets the shared resolution visit counter.
        authentication = _TypeBuilder(self.analysis)
        for scope, original, types in receipt.scopes:
            _validate_scope(self.analysis, scope)
            self.resolution_scopes[id(scope)] = scope
            if authentication.scope(scope) is not original:
                raise CompilationError("descriptor metadata declaring scope authority changed")
            current = getattr(scope, "component_types", {})
            if set(current) != {name for name, *_ in types}:
                raise CompilationError("descriptor metadata original type membership changed")
            for name, schema, private, reason in types:
                old = original[5].get(name)
                if (old is None or current[name] is not schema or old[0] is not schema
                        or schema.scope is not scope or schema.name != name or old[1] is not schema.node
                        or schema.private != private or schema.reason != reason
                        or receipt.session is None and (old[2] != str(schema.node) or old[3] != _members(schema))
                        or receipt.session is not None and not receipt.session.contains(schema)):
                    raise CompilationError("descriptor metadata original type schema changed")
        _validate_scope_roles(self.analysis, self.scope_roles, self.resolution_scopes)
        schema = receipt.schema
        if getattr(schema.scope, "component_types", {}).get(schema.name) is not schema or schema.reason:
            raise CompilationError("descriptor metadata resolved type identity changed")
        self.metadata_schemas[id(binding)] = receipt
        return schema

    def scalar_metadata_call(self, routine, call):
        """Prove descriptor preservation, without mapping record payloads.

        Ordinary source-call mapping deliberately cannot capture fields of an
        array of records. Original type association can still establish a
        scalar-field native call, optionally with independent fixed local
        vector companions, whose complete original closure is checked.
        """
        if _kind(call) != "Call_Stmt":
            return False
        from compiler.frontend.native_metadata import access as metadata_access

        scope = self.analysis.source_scope_for(call, routine.scope)
        fields = []
        for node in _walk(call.items[1]):
            if _kind(node) == "Data_Ref" and _kind(node.items[0]) == "Part_Ref":
                access = metadata_access(self.analysis, scope, node)
                self.metadata_object(access.root_binding, access)
                fields.append(access)
        if not fields:
            return False
        self.resolution(routine, call)
        resolved = resolve_original_call(self.analysis, scope, call)
        for owner in (routine, resolved.callee):
            specification = next((part for part in _children(owner.scope.node)
                                  if _kind(part) == "Specification_Part"), None)
            if any(_kind(node) in {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt"}
                   for node in _walk(specification)):
                raise DescriptorWindowUnsupported("descriptor metadata call has unproved declaration association")
        vectors = []
        for association in resolved.associations:
            formal, actual, typed = association.formal_binding, association.actual, association.actual_type
            if (actual is None or typed is None
                    or formal.dtype not in {"integer", "real", "logical"}
                    or formal.attributes & (_UNSAFE | {"allocatable", "target"})
                    or typed.binding is not None and typed.binding.attributes & _UNSAFE):
                raise DescriptorWindowUnsupported("descriptor metadata call requires independent numeric scalar actuals")
            if formal.rank or typed.rank:
                binding = typed.binding
                if (formal.rank != 1 or typed.rank != 1 or _kind(actual) != "Name" or binding is None
                        or binding.declaring_scope is not routine.scope or binding.name in routine.arguments
                        or binding.intent is not None or binding.attributes & {"allocatable", "target", "parameter"}):
                    raise DescriptorWindowUnsupported("descriptor metadata vector requires independent fixed local storage")
                formal_bounds = self.fixed_vector_bounds(resolved.callee, formal)
                actual_bounds = self.fixed_vector_bounds(routine, binding)
                extent = max(0, formal_bounds[1] - formal_bounds[0] + 1)
                if extent != max(0, actual_bounds[1] - actual_bounds[0] + 1):
                    raise DescriptorWindowUnsupported("descriptor metadata vector shapes do not conform")
                if any(other is not association and other.actual_type is not None
                       and other.actual_type.binding is binding for other in resolved.associations):
                    raise DescriptorWindowUnsupported("descriptor metadata vector argument storage overlaps another argument")
                vectors.append({"formal": association.formal, "formal_bounds": list(formal_bounds),
                    "actual_binding": binding.public(), "actual_bounds": list(actual_bounds), "extent": extent,
                    "descriptor_evidence_only": True, "payload_access_authorized": False,
                    "entry_definition_authority": False, "execution_authorized": False})
            self.expression(routine, actual)
        self.operations += 1
        self.metadata_calls[id(call)] = {
            "association": resolved.public(), "descriptor_evidence_only": True,
            "scalar_fields": [list(access.path) for access in fields],
            "whole_original_closure_required": True, "capture_authorized": False,
            "payload_access_authorized": False, "entry_definition_authority": False,
            "execution_authorized": False}
        if vectors:
            self.metadata_calls[id(call)]["fixed_vector_arguments"] = vectors
        nested = (routine.qualified, id(call)) in self.team_calls
        self.team_call_depth += int(nested)
        try:
            self.procedure(resolved.procedure)
        finally:
            self.team_call_depth -= int(nested)
        return True

    def fixed_vector_bounds(self, routine, binding):
        if len(binding.shape_nodes) != 1 or _kind(binding.shape_nodes[0]) != "Explicit_Shape_Spec":
            raise DescriptorWindowUnsupported("descriptor metadata vector requires one fixed explicit shape")
        # Original constant expressions retain the ordinary source lookup and
        # syntax charges. No reached descriptor or payload is evaluated here.
        self.resolution(routine, binding.shape_nodes[0])
        try:
            return self.object_builder.bounds(binding.declaring_scope, binding)[0]
        except CompilationError as error:
            raise DescriptorWindowUnsupported("descriptor metadata vector bounds require exact INTEGER constants") from error

    def metadata_object(self, binding, access=None):
        """Authenticate storage/type only, not the layout of a whole record."""
        from compiler.frontend.component_bindings import _typename

        if (binding.declaring_scope is None or not binding.rank or not binding.dtype.startswith("type(")
                or binding.attributes & (_UNSAFE | {"target"})):
            raise DescriptorWindowUnsupported("descriptor metadata requires an independent original array object")
        declaration = self.object_builder.scope(binding.declaring_scope)
        specification = next((part for part in _children(binding.declaring_scope.node)
                              if _kind(part) == "Specification_Part"), None)
        for node in _walk(specification):
            self.tick()
            if (_kind(node) in {"Common_Stmt", "Equivalence_Stmt", "Bind_Stmt", "Cray_Pointer_Stmt",
                                "Namelist_Stmt", "Codimension_Attr_Spec"}
                    or _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")):
                raise DescriptorWindowUnsupported("descriptor metadata object has unproved declaration association")
        original = declaration[4].get(binding.name)
        if (original is None or original[0] is not binding
                or original[1] != _canonical(binding.public())):
            raise CompilationError("descriptor metadata object changed its original declaration")
        schema = self.metadata_schema(binding, _typename(binding.dtype))
        if id(binding) not in self.metadata_objects:
            self.metadata_objects[id(binding)] = {
                "version": 1, "original_binding": binding.public(),
                "declaring_type": {"module": schema.scope.module, "name": schema.name},
                "type_definition_sha256": sha256(str(schema.node).encode()).hexdigest(),
                "selected_fields": {}, "descriptor_evidence_only": True,
                "original_reached_selector_required": True,
                "whole_object_layout_authorized": False, "whole_object_copy_authorized": False,
                "capture_authorized": False, "entry_definition_authority": False,
                "payload_access_authorized": False, "execution_authorized": False}
        if access is not None:
            self.metadata_objects[id(binding)]["selected_fields"]["%".join(access.path)] = {
                "component_path": list(access.path), "original_field": access.binding.public(),
                "scalar_numeric_storage_only": True, "allocation_or_pointer_component": False}

    def fixed_object(self, binding):
        from compiler.frontend.library_storage_calls import library_storage_declaration

        routine = self.analysis.routines.get(getattr(binding.declaring_scope, "qualified", None))
        private = library_storage_declaration(self.analysis, routine, binding) if routine is not None else None
        if private is not None:
            self.library_states[id(binding)] = private
            return private._declaration.public()
        if id(binding) not in self.objects:
            schema = self.object_builder.binding(binding)
            if id(schema.node) not in self.object_schemas:
                self.object_schemas[id(schema.node)] = self.object_builder.schema(schema)
            self.objects[id(binding)] = {
                "original_binding": binding.public(),
                "type_definition": self.object_schemas[id(schema.node)],
                "descriptor_evidence_only": True,
                "native_continuation_authority": False, "numerical_abi": False}
        return self.objects[id(binding)]

    def joined_groups(self, routine, graph, selected):
        roots = tuple(node for identity in selected for node in graph.source_nodes(identity))
        present = set()
        for original in _walk(roots):
            self.tick()
            present.add(id(original))

        def discover(items, peeled):
            for item in grouped_original_sequence(items, peeled):
                self.tick()
                if isinstance(item, NativeGroupCandidate):
                    from compiler.frontend.descriptor_teams import _original_group

                    # Graph selection omits ordinary documentation comments.
                    # Recover the genuine full original tuple before asking
                    # any independent joined issuer to authenticate it.
                    item = _original_group(routine, item.original_nodes, graph_selection=True)
                    # Membership is a control restriction, independent of
                    # whether joined or native-prefix authority checks the
                    # directives. Neither proof permits exiting an OMP team.
                    for original in _walk(item.original_nodes):
                        self.tick()
                        if _kind(original) == "Return_Stmt":
                            self.joined_returns.add((routine.qualified, id(original)))
                        elif _kind(original) in {"Nonlabel_Do_Stmt", "Exit_Stmt", "Cycle_Stmt"}:
                            self.joined_loop_controls.add((routine.qualified, id(original)))
                    try:
                        self.join_analysis.joined_completion(routine.qualified, item.original_nodes)
                    except CompilationError:
                        # Missing authority remains an error when its
                        # original directive is visited; an independently
                        # registered native-prefix proof remains usable.
                        try:
                            self.join_analysis.numerical_joined_completion(routine.qualified, item.original_nodes)
                        except CompilationError:
                            # Storage structure is independent of numerical
                            # execution legality. The whole original body and
                            # call closure are still rescanned below.
                            with suppress(CompilationError):
                                self.storage_teams.append(prove_descriptor_team(
                                    self.join_analysis, routine.qualified, item.original_nodes))
                elif _kind(item) in {"Execution_Part", "If_Construct", "Block_Nonlabel_Do_Construct",
                                     "Associate_Construct"}:
                    discover(peeled.get(id(item), _children(item)), peeled)

        discover(roots, {})
        proofs = (*self.join_analysis._joined_completions.values(),
                  *self.join_analysis._numerical_completions.values(), *self.storage_teams)
        for proof in proofs:
            if not isinstance(proof, (NativeCompletionProof, NumericalCompletionProof, DescriptorTeamProof)) or proof.procedure != routine.qualified:
                continue
            if isinstance(proof, (NativeCompletionProof, DescriptorTeamProof)):
                proof.validate(self.join_analysis, routine.qualified, proof.selected_node_ids)
            else:
                proof.validate(self.join_analysis)
            originals = tuple(node for identity in proof.selected_node_ids for node in graph.source_nodes(identity))
            if not all(id(node) in present for node in originals):
                continue
            self.team_calls.update((routine.qualified, id(node)) for node in _walk(originals)
                                   if _kind(node) == "Call_Stmt")
            for original in _walk(originals):
                self.tick()
                if _kind(original) == "Comment" and str(original).lstrip().lower().startswith("!$omp"):
                    self.joined_directives[(routine.qualified, id(original))] = proof

    def directive(self, routine, original):
        proof = self.joined_directives.get((routine.qualified, id(original)))
        if proof is not None:
            if isinstance(proof, (NativeCompletionProof, DescriptorTeamProof)):
                proof.validate(self.join_analysis, routine.qualified, proof.selected_node_ids)
            else:
                proof.validate(self.join_analysis)
            self.joins[proof.identity] = {
                "proof_identity": proof.identity, "procedure": proof.procedure,
                "structured_identity": proof.structured_identity,
                "selected_node_ids": list(proof.selected_node_ids),
                "proof_role": ("native_joined_completion" if isinstance(proof, NativeCompletionProof)
                               else "original_joined_storage_structure" if isinstance(proof, DescriptorTeamProof)
                               else "numerical_source_completion"),
                "descriptor_evidence_only": True, "whole_original_group_scanned": True,
                "retains_original_team_and_directives": True,
                "internal_cuts_authorized": False, "execution_authorized": False,
                "entry_definition_authority": False, "gpu_legality_established": False}
            if isinstance(proof, DescriptorTeamProof):
                self.joins[proof.identity]["storage_structure"] = proof.public()
            return
        if routine.qualified not in self.prefix_scopes and self.worksharing_directive(routine, original):
            return
        self.prefix_directive(routine)

    def worksharing_directive(self, routine, original):
        from compiler.frontend.descriptor_worksharing import (
            WorksharingStorageUnsupported,
            original_worksharing_storage,
        )

        # This premise is restricted to a whole original called procedure.
        # It never grants participation or completion at a selected root.
        if len(self.active) < 2 or self.active[-1] != routine.qualified:
            return False
        if routine.qualified not in self.worksharing_storage:
            try:
                proof = original_worksharing_storage(self.analysis, routine)
            except WorksharingStorageUnsupported as error:
                raise DescriptorWindowUnsupported(str(error)) from error
            self.worksharing_storage[routine.qualified] = proof, proof.checkpoint()
            for opening, _loop, ending, _collapse in proof._regions:
                self.worksharing_directives[routine.qualified, id(opening)] = proof
                self.worksharing_directives[routine.qualified, id(ending)] = proof
        proof, baseline = self.worksharing_storage[routine.qualified]
        proof.require_checkpoint(baseline)
        if self.worksharing_directives.get((routine.qualified, id(original))) is not proof:
            raise DescriptorWindowUnsupported("descriptor worksharing directive exceeds its exact original envelope")
        return True

    def prefix_directive(self, routine):
        proof = self.prefix_scopes.get(routine.qualified)
        if proof is None:
            raise DescriptorWindowUnsupported("descriptor window original OpenMP storage requires registered whole-closure authority")
        proof.validate(self.analysis, proof.procedure)
        public = proof.public()
        self.prefixes[proof.identity] = {
            "proof_identity": proof.identity, "procedure": proof.procedure,
            "covered_original_procedures": [item["procedure"] for item in public["closure"]],
            "descriptor_evidence_only": True,
            "whole_original_closure_scanned": True,
            "native_prefix_execution_requirements": {
                "requires_original_native_execution": public["requires_original_native_execution"],
                "requires_serial_caller": public["requires_serial_caller"],
                "requires_no_active_context": public["requires_no_active_context"]},
            "execution_authorized": False, "allocation_freshness_authority": False,
            "entry_definition_authority": False, "gpu_legality_established": False}

    def tick(self):
        self.nodes += 1
        if self.nodes > self.source_budget.limit:
            raise CompilationError("descriptor window original-source traversal budget exhausted")

    def resolution_tick(self, role):
        self.resolution_work += 1
        if role == "syntax":
            self.resolution_syntax += 1
        else:
            self.resolution_lookups += 1
        self.tick()

    def resolution(self, routine, originals, *, names=()):
        # Each original semantic unit retains the lookup/depth cap. Both its
        # syntax and resolution transitions spend the existing overall source
        # scan budget; splitting units cannot hide aggregate unbounded work.
        self.resolution_units += 1
        _validate_resolution(self.analysis, routine.scope, _walk(originals), tick=self.resolution_tick, names=names,
                             _scope_validator=self.scope_authority)

    def scope_authority(self, scope):
        # Retained exact objects, local to this one original-analysis query.
        # Lookup transitions and their per-unit limits are never memoized.
        if id(scope) not in self.resolution_scopes:
            _validate_scope(self.analysis, scope)
            self.resolution_scopes[id(scope)] = scope

    def publish(self, procedure, authority):
        private_allocations = tuple(self.private_allocations.items())
        worksharing = tuple(self.worksharing_storage.items())
        from compiler.frontend.descriptor_associations import _attached as _same_fields
        from compiler.frontend.descriptor_associations import _attachments, _same_syntax

        numeric_reads = tuple(self.numeric_component_reads.items())
        intrinsic_tokens = tuple(self.intrinsic_tokens.items())
        intrinsic_authority = self.intrinsic_authority
        joined_blocks = tuple(self.joined_blocks.items())
        numeric_roots = tuple((receipt[2], _attachments(receipt[2]), receipt[2].declaring_scope)
                              for _key, receipt in numeric_reads)
        associations = tuple((key, proof, tuple((part.name, getattr(proof, part.name))
                              for part in fields(proof)), proof.public())
                             for key, proof in self.fixed_associations.items())
        for _name, (proof, baseline) in worksharing:
            proof.require_checkpoint(baseline)
            proof.validate(self.analysis, proof._routine)
        for _key, (proof, baseline) in private_allocations:
            proof.require_checkpoint(baseline)
            proof.validate(self.join_analysis, proof._routine, proof.binding)
        for _key, (proof, baseline) in joined_blocks:
            proof.require_checkpoint(baseline)
            proof.validate(scope_validator=self.scope_authority, tick=self.tick)
        for _key, proof, _attached, _public in associations:
            proof.validate(scope_validator=self.scope_authority, tick=self.tick)
        for proof in self.record_calls.values():
            proof.validate()
        for proof in self.library_states.values():
            proof.validate()
        for proof, storage in self.library_calls.values():
            proof.validate()
            storage.validate(self.analysis, proof._routine.qualified)
        for key, (routine, graph, group, proof) in self.storage_group_tokens.items():
            proof.validate()
            if (self.analysis.structure(routine.qualified) is not graph
                    or graph.native_groups.get(key[1]) is not group
                    or not group.matches(proof._group.original_nodes)
                    or self.storage_groups.get(key) != {"procedure": routine.qualified,
                        "source_graph": graph.identity, "original_node_id": key[1], "storage_group": proof.public()}):
                raise CompilationError("descriptor group source or independent storage authority changed at publication")
        for routine, call, proof in self.scalar_clock_tokens.values():
            proof.validate(self.join_analysis, routine.qualified, call)
            public = proof.public()
            if (_hash({key: value for key, value in public.items() if key != "identity"}) != proof.identity
                    or self.scalar_clocks.get(proof.identity) != {
                        **public, "descriptor_evidence_only": True, "execution_authorized": False}):
                raise CompilationError("descriptor scalar clock changed before source publication")
        self.source_budget.validate()
        # A later syntax/lookup tick may mutate a previously checked scope.
        # No charged callbacks follow this fresh publication fence.
        for receipt in self.metadata_schemas.values():
            self.metadata_schema(receipt.binding, receipt.type_name)
        # Constant bounds consume the declaring namespace and kind environment,
        # not only the resulting integer. Recheck them after all charged work.
        fixed_types = _TypeBuilder(self.analysis)
        for scope in self.fixed_module_scopes.values():
            fixed_types.scope(scope)
        for binding, module, axes, declaration in self.fixed_module_bindings.values():
            if (module.bindings.get(binding.name) is not binding or binding.declaring_scope is not module
                    or _canonical(binding.public()) != declaration or len(binding.shape_nodes) != len(axes)
                    or any(a is not b for a, b in zip(binding.shape_nodes, axes, strict=True))):
                raise CompilationError("fixed module consumed declaration or descriptor changed at publication")
        self.validate_numeric_component_reads()
        self.validate_native_intrinsics(intrinsic_tokens)
        _validate_scope_roles(self.analysis, self.scope_roles, self.resolution_scopes)
        for scope in self.resolution_scopes.values():
            _validate_scope(self.analysis, scope)
        syntax = self.analysis._require_original_global()
        for name in (procedure, *self.checked):
            self.analysis._require_original_routine(name, _syntax=syntax)
        if _hash(self.analysis._summary_authority(_syntax=syntax)) != authority:
            raise CompilationError("descriptor window original source authority changed")
        # The final authority callback must not publish a proof over source
        # bytes changed after the preceding original-syntax/session check.
        # Any mutations during file verification still face the cheap retained
        # declaration, namespace and registry checks below.
        self.analysis.inputs.verify_live()
        if (len(self.fixed_associations) != len(associations)
                or any(self.fixed_associations.get(key) is not proof or proof.public() != public
                       or any(getattr(proof, name) is not value for name, value in attached)
                       for key, proof, attached, public in associations)):
            raise CompilationError("descriptor association source attachment changed at publication")
        if (len(self.numeric_component_reads) != len(numeric_reads)
                or any(self.numeric_component_reads.get(key) is not receipt
                       or not _same_syntax(receipt[6]) for key, receipt in numeric_reads)
                or any(not _same_fields(root, attached) or root.declaring_scope is not scope
                       for root, attached, scope in numeric_roots)):
            raise CompilationError("descriptor numeric component read attachment changed at final publication")
        if (len(self.intrinsic_tokens) != len(intrinsic_tokens)
                or len(self.intrinsics) != len(intrinsic_tokens)):
            raise CompilationError("descriptor intrinsic consumed authority inventory changed at final publication")
        self.validate_native_intrinsics(intrinsic_tokens, rederive=False, authority=intrinsic_authority)
        if (len(self.joined_blocks) != len(joined_blocks)
                or any(self.joined_blocks.get(key) is not receipt for key, receipt in joined_blocks)):
            raise CompilationError("descriptor joined BLOCK inventory changed at final publication")
        for _key, (proof, baseline) in joined_blocks:
            proof.require_checkpoint(baseline)
        if (len(self.private_allocations) != len(private_allocations)
                or any(self.private_allocations.get(key) is not receipt
                       for key, receipt in private_allocations)):
            raise CompilationError("descriptor private allocation obligations changed at publication")
        if (len(self.worksharing_storage) != len(worksharing)
                or any(self.worksharing_storage.get(name) is not receipt for name, receipt in worksharing)):
            raise CompilationError("descriptor worksharing retained inventory changed at publication")
        for _name, (proof, baseline) in worksharing:
            proof.require_checkpoint(baseline)
        for _key, (proof, baseline) in private_allocations:
            proof.require_checkpoint(baseline)
        if self.private_allocation_source is not None:
            from compiler.frontend.local_allocatable_storage import require_allocation_source_checkpoint

            require_allocation_source_checkpoint(self.analysis, self.private_allocation_source)

    def declaration_resolution(self, routine):
        specification = next((part for part in _children(routine.scope.node)
                              if _kind(part) == "Specification_Part"), None)
        declarations = (declaration for part in _children(specification)
                        for declaration in (_children(part) if _kind(part) == "Implicit_Part" else (part,)))
        for declaration in declarations:
            self.tick()
            if _kind(declaration) == "Type_Declaration_Stmt":
                dtype, _attributes, entities = declaration.items
                selector = dtype.items[1] if _kind(dtype) == "Intrinsic_Type_Spec" else None
                if selector is not None:
                    self.resolution(routine, selector)
                for entity in _children(entities):
                    self.tick()
                    if _kind(entity) == "Entity_Decl" and entity.items[3] is not None:
                        self.resolution(routine, entity.items[3])
            elif _kind(declaration) == "Parameter_Stmt":
                for definition in _children(declaration.items[1]):
                    self.tick()
                    if _kind(definition) == "Named_Constant_Def":
                        self.resolution(routine, definition.items[1])

    def expression(self, routine, node, depth=0, *, descriptor=False):
        from compiler.frontend.descriptor_expression import visit_descriptor_expression

        # Kept for existing diagnostic callers; procedure() separately bounds
        # call recursion, while source arithmetic/control consume syntax work.
        del depth
        visit_descriptor_expression(self, routine, node, descriptor=descriptor)

    def expression_children(self, routine, node, *, descriptor=False):
        self.tick()
        if not isinstance(node, (Base, tuple, list)):
            return
        kind = _kind(node)
        scope = self.analysis.source_scope_for(node, routine.scope)
        if (kind in {"Proc_Component_Ref", "Coindexed_Named_Object"}
                or kind.startswith("Defined_") and kind.endswith("_Op")):
            raise DescriptorWindowUnsupported("descriptor window association/operator effects require separate authority")
        if kind == "Data_Ref":
            if _kind(node.items[0]) == "Part_Ref":
                from compiler.frontend.native_metadata import access as metadata_access

                access = metadata_access(self.analysis, scope, node)
                self.metadata_object(access.root_binding, access)
            else:
                try:
                    access = component_access(self.analysis, scope, node)
                except ComponentStorageUnsupported:
                    self.numeric_component_read(routine, node)
                    return
                original_root = access.root_binding
                base = self.analysis._binding(scope, node.items[0])
                for proof in reversed(self.association_stack):
                    declaration = proof._association
                    matched = [item for name, item in zip(declaration.scope.bindings, declaration.selectors, strict=True)
                               if declaration.scope.bindings[name] is base]
                    if matched:
                        original_root = matched[0].root_binding
                        break
                self.fixed_object(original_root)
            if access.binding.dtype not in {"real", "integer", "logical"}:
                raise DescriptorWindowUnsupported("descriptor window whole-object effects require separate authority")
            for index in access.indices:
                yield index, False
            return
        if kind == "Actual_Arg_Spec":
            yield node.items[1], descriptor
            return
        if kind == "Name":
            binding = self.analysis._binding(scope, node)
            if binding is not None and binding.dtype.startswith("type(") and descriptor:
                self.metadata_object(binding)
                return
            if self.optional_output(routine, binding):
                from compiler.frontend.descriptor_optional import original_positive_presence

                if not original_positive_presence(self, routine, node, binding):
                    raise DescriptorWindowUnsupported("descriptor optional output requires original positive PRESENT presence")
                return
            if (binding is None or binding.attributes & _UNSAFE and not self.optional_scalar(routine, binding)
                    or binding.dtype.startswith("type(")):
                raise DescriptorWindowUnsupported("descriptor window original expression storage is unproved: " + str(node))
            return
        if kind in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference", "Structure_Constructor"}:
            name, args = node.items
            binding = self.analysis._binding(scope, name)
            predicates = [proof for proof in self.analysis._native_predicates.values()
                          if isinstance(proof, NativePredicateProof) and proof.procedure == routine.qualified
                          and proof._expression is node]
            if predicates:
                if len(predicates) != 1:
                    raise CompilationError("descriptor window requires unique registered original native predicate proof")
                from compiler.frontend.source_functions import _origin

                self.retain_expression(predicates[0], _origin(self.analysis), "_native_predicates", routine)
                proof = predicates[0].validate(self.analysis, routine.qualified, node)
                self.predicates[proof.identity] = proof.public()
                for actual in proof.actuals:
                    yield actual, False
                return
            if kind == "Part_Ref" and binding is not None and binding.rank:
                yield name, False
                yield args, False
                return
            candidates = self.analysis._candidates(scope, name)
            if candidates:
                proofs = [proof for proof in self.analysis._native_functions.values()
                          if proof.procedure == routine.qualified and proof._expression is node]
                metadata = False
                if not proofs:
                    from compiler.frontend.descriptor_array_functions import DescriptorArraySourceFunctionProof
                    from compiler.frontend.source_functions import DescriptorSourceFunctionProof, _origin

                    proofs = [proof for proof in _origin(self.analysis)._descriptor_functions.values()
                              if type(proof) in {DescriptorSourceFunctionProof, DescriptorArraySourceFunctionProof}
                              and proof.procedure == routine.qualified and proof._expression is node]
                    metadata = True
                if len(proofs) != 1:
                    if proofs:
                        raise CompilationError("descriptor window requires unique registered original source-function proof")
                    from compiler.frontend.storage_mutation_effects import StorageMutationEffects

                    projections = [proof for proof in getattr(_origin(self.analysis), "_storage_mutation_effects", {}).values()
                                   if type(proof) is StorageMutationEffects and proof._procedure == routine.qualified
                                   and proof._expression is node]
                    if not projections:
                        array_demand = any(
                            binding.rank and any(_kind(axis) == "Assumed_Shape_Spec" for axis in binding.shape_nodes)
                            for candidate in candidates if candidate in self.analysis.numerical_helpers
                            for name, binding in self.analysis.numerical_helpers[candidate].scope.bindings.items()
                            if name in self.analysis.numerical_helpers[candidate].arguments)
                        legacy_eligible = len(self.active) >= 2 and (
                            any(frame[4] in {"bare", "while"} for frame in self.loop_frames)
                            or array_demand and self.active[-1] == routine.qualified)
                        demand = _DescriptorFunctionRequired(routine.qualified, node, legacy_eligible=legacy_eligible)
                        if self.function_demands is None:
                            raise demand
                        key = routine.qualified, id(node)
                        if key not in self.function_demands:
                            if len(self.function_demands) >= self.analysis.procedure_limit:
                                raise CompilationError("descriptor source function admission request budget exhausted")
                            self.function_demands[key] = demand
                        # This private discovery pass cannot classify the
                        # unknown function or its actuals. It must be discarded
                        # before publication; the next complete pass checks both.
                        return
                    if len(projections) != 1:
                        raise CompilationError("descriptor window requires unique registered original storage projection")
                    projection = projections[0]
                    closure = _raw_value(projection, "_record")["closure"]
                    self.retain_expression(projection, _origin(self.analysis), "_storage_mutation_effects", routine,
                                           tuple(member["procedure"] for member in closure))
                    proof = projection
                    if not proof.preserves_descriptors(_origin(self.analysis), routine.qualified, node):
                        raise DescriptorWindowUnsupported("descriptor window source-function storage mutation/escape closure is unavailable")
                    self.storage_functions[proof.identity] = proof.public()
                    # The original actuals still pass this window's typed
                    # descriptor classifier. The storage token never supplies
                    # payload access, initialization, or native continuation.
                    if _raw_value(proof, "_record").get("readonly_record_associations"):
                        for actual in proof.descriptor_actuals(routine, node):
                            yield actual, False
                    else:
                        yield args, False
                    return
                from compiler.frontend.source_functions import _origin

                self.retain_expression(proofs[0], _origin(self.analysis),
                                       "_descriptor_functions" if metadata else "_native_functions", routine,
                                       (_raw_value(proofs[0], "callee"),))
                proof = proofs[0].validate(self.analysis, routine.qualified, node,
                                           resolution_tick=self.resolution_tick)
                (self.descriptor_functions if metadata else self.functions)[proof.identity] = proof.public()
                if metadata and type(proof) is DescriptorArraySourceFunctionProof:
                    for actual in proof.resolved.actuals:
                        yield actual, False
                else:
                    for mapping in proof.resolved.mappings:
                        yield mapping.actual, False
                return
            if kind == "Intrinsic_Function_Reference" and str(name).lower() == "norm2":
                from compiler.frontend.descriptor_intrinsics import norm2_descriptor_effects

                actuals = self.native_intrinsic(routine, node, norm2_descriptor_effects)
                for actual in actuals:
                    yield actual, False
                return
            if kind == "Intrinsic_Function_Reference" and str(name).lower() in {"matmul", "transpose"}:
                from compiler.frontend.descriptor_intrinsics import matrix_descriptor_effects

                actuals = self.native_intrinsic(routine, node, matrix_descriptor_effects)
                for actual in actuals:
                    yield actual, False
                return
            if (kind != "Intrinsic_Function_Reference" or binding is not None
                    or self.analysis._unknown_exports(scope) or str(name).lower() not in _INTRINSICS):
                raise DescriptorWindowUnsupported("descriptor window unknown function/escape effects: " + str(name))
            if str(name).lower() in ARRAY_INQUIRIES | {"allocated", "present"}:
                for index, argument in enumerate(_children(args)):
                    yield argument, index == 0
            else:
                yield args, False
            return
        for child in _children(node):
            yield child, False

    def loop_frame(self, routine, graph, identity):
        originals = graph.source_nodes(identity)
        if len(originals) != 1 or _kind(originals[0]) != "Block_Nonlabel_Do_Construct":
            raise DescriptorWindowUnsupported("descriptor window loop requires an exact nonlabelled original construct")
        original = originals[0]
        header, ending = original.content[0], original.content[-1]
        if _kind(header) != "Nonlabel_Do_Stmt" or _kind(ending) != "End_Do_Stmt":
            raise DescriptorWindowUnsupported("descriptor window loop requires its exact original header and end")
        control = header.items[1]
        if control is None:
            mode, condition = "bare", None
        elif len(control.items) != 4 or control.items[3] is not None:
            raise DescriptorWindowUnsupported("descriptor window concurrent or unknown loop control requires separate authority")
        elif control.items[0] is not None and control.items[1] is None:
            mode, condition = "while", control.items[0]
        elif control.items[0] is None and control.items[1] is not None:
            mode, condition = "counted", None
        else:
            raise DescriptorWindowUnsupported("descriptor window unknown original loop control")
        name = getattr(getattr(header, "item", None), "name", None)
        name = str(name).lower() if name else None
        closing = str(ending.items[1]).lower() if ending.items[1] is not None else None
        if closing is not None and closing != name:
            raise DescriptorWindowUnsupported("descriptor window original loop construct name mismatch")
        if mode != "counted":
            if (len(self.active) < 2 or self.active[-1] != routine.qualified
                    or (routine.qualified, id(header)) in self.joined_loop_controls):
                raise DescriptorWindowUnsupported("descriptor window non-counted loop requires a complete native callee outside original OpenMP")
            self.native_loops[routine.qualified, identity] = {
                "procedure": routine.qualified, "source_graph": graph.identity, "original_node_id": identity,
                "mode": mode, "construct_name": name, "condition": str(condition) if condition is not None else None,
                "complete_original_body_scanned": True, "descriptor_evidence_only": True,
                "termination_authority": False, "execution_authorized": False,
                "entry_definition_authority": False, "payload_access_authorized": False}
        return routine.qualified, identity, header, name, mode, condition

    def loop_control(self, routine, graph, identity, original):
        if len(self.active) < 2 or self.active[-1] != routine.qualified:
            raise DescriptorWindowUnsupported("descriptor window loop exit requires a complete original native callee")
        name = str(original.items[1]).lower() if original.items[1] is not None else None
        target = next((frame for frame in reversed(self.loop_frames)
                       if frame[0] == routine.qualified and (name is None or frame[3] == name)), None)
        if target is None:
            raise DescriptorWindowUnsupported("descriptor window loop exit requires its exact same-callee lexical target")
        if ((routine.qualified, id(original)) in self.joined_loop_controls
                or (routine.qualified, id(target[2])) in self.joined_loop_controls):
            raise DescriptorWindowUnsupported("descriptor window loop exit cannot cross original OpenMP control")
        self.resolution(routine, original)
        self.operations += 1
        self.native_loop_controls[routine.qualified, identity] = {
            "procedure": routine.qualified, "source_graph": graph.identity, "original_node_id": identity,
            "original_statement": str(original), "target_loop_id": target[1], "target_construct_name": target[3],
            "complete_original_body_scanned": True, "descriptor_evidence_only": True,
            "termination_authority": False, "execution_authorized": False,
            "entry_definition_authority": False, "payload_access_authorized": False}

    def visit(self, routine, graph, identities, depth=0):
        from compiler.frontend.descriptor_control import visit_descriptor_control

        del depth  # Source nesting is not a procedure-call recursion depth.
        visit_descriptor_control(self, routine, graph, identities)

    def control_children(self, routine, graph, identities):
        for identity in identities:
            self.tick()
            node = graph.nodes[identity]
            if node.kind == "sequence":
                yield node.children
            elif node.kind == "loop":
                self.loop_frames.append(self.loop_frame(routine, graph, identity))
                try:
                    yield node.children
                finally:
                    self.loop_frames.pop()
            elif node.kind == "branch":
                for condition, body in node.alternatives:
                    yield (body,) if condition is None else (condition, body)
            elif node.kind == "associate":
                from compiler.frontend.descriptor_associations import original_fixed_association

                original, = graph.source_nodes(identity)
                proof = original_fixed_association(self.analysis, routine, original,
                    scope_validator=self.scope_authority, tick=self.tick)
                self.fixed_associations[routine.qualified, identity] = proof
                # The typed declaration receipt covers only the exact selector.
                # Every original body node retains its ordinary effect checks.
                if len(node.children) != 2 or graph.nodes[node.children[0]].details.get("evaluation") != "associate_selector":
                    raise DescriptorWindowUnsupported("descriptor association changed its original selector/body graph")
                self.operations += 1
                self.association_stack.append(proof)
                try:
                    yield node.children[1:]
                finally:
                    self.association_stack.pop()
            elif node.kind == "operation":
                self.operations += 1
                role = node.details.get("evaluation")
                for original in graph.source_nodes(identity):
                    if role == "directive":
                        self.directive(routine, original)
                    elif role == "condition":
                        self.resolution(routine, original)
                        self.expression(routine, original)
                    elif role == "loop_header":
                        control = original.items[1]
                        if control is None or control.items[1] is None:
                            frame = self.loop_frames[-1] if self.loop_frames else None
                            if frame is None or frame[0] != routine.qualified or frame[2] is not original:
                                raise DescriptorWindowUnsupported("descriptor window non-counted header requires its exact complete original loop")
                            if frame[4] == "while":
                                self.resolution(routine, frame[5])
                                self.expression(routine, frame[5])
                        else:
                            self.resolution(routine, control.items[1])
                            self.expression(routine, control.items[1])
                    elif _kind(original) == "Assignment_Stmt":
                        if self.library_constructor(routine, original):
                            continue
                        target, _equals, value = original.items
                        self.resolution(routine, (target, value))
                        scope = self.analysis.source_scope_for(original, routine.scope)
                        binding = self.analysis._binding(scope, target) if _kind(target) == "Name" else None
                        if (binding is not None and "allocatable" in binding.attributes
                                and not self.private_allocation_mutation(routine, binding)):
                            from compiler.frontend.descriptor_scalar_fill import (
                                ScalarFillUnsupported,
                                scalar_fill_descriptor_effects,
                            )

                            try:
                                self.native_intrinsic(routine, original, scalar_fill_descriptor_effects,
                                                      names=("assignment(=)",))
                            except ScalarFillUnsupported as error:
                                raise DescriptorWindowUnsupported(
                                    "descriptor window whole allocatable assignment may replace storage") from error
                        self.expression(routine, target)
                        self.expression(routine, value)
                    elif _kind(original) != "Continue_Stmt":
                        raise DescriptorWindowUnsupported("descriptor window operation lacks descriptor/escape authority")
            elif node.kind == "call":
                self.operations += 1
                originals = graph.source_nodes(identity)
                if len(originals) != 1 or node.details.get("contract"):
                    raise DescriptorWindowUnsupported("descriptor window requires original direct source calls")
                call = originals[0]
                if self.library_call(routine, graph, identity, call):
                    continue
                if self.scalar_clock(routine, call):
                    continue
                self.resolution(routine, call)
                if self.terminating_call(routine, call):
                    continue
                if self.owned_record_call(routine, call):
                    continue
                scope = self.analysis.source_scope_for(call, routine.scope)
                resolved = resolve_source_call(self.analysis, scope, call)
                for mapping in resolved.mappings:
                    if self.optional_scalar_mapping(routine, call, resolved, mapping) is not None:
                        if (mapping.presence == "forwarded_optional"
                                and self.optional_output(self.analysis.routines[resolved.procedure], mapping.formal_binding)):
                            # Exact original name forwarding does not evaluate
                            # the optional payload; every callee use is still
                            # visited and needs its own original PRESENT guard.
                            self.tick()
                        else:
                            self.expression(routine, mapping.actual)
                        continue
                    if (mapping.presence != "supplied"
                            or mapping.formal_binding.attributes & _UNSAFE
                            or mapping.binding is not None and mapping.binding.attributes & _UNSAFE):
                        raise DescriptorWindowUnsupported("descriptor window call association/escape authority is unavailable")
                    if mapping.source_object is not None:
                        self.fixed_object(mapping.formal_binding)
                        self.fixed_object(mapping.binding)
                        self.object_mappings[mapping.source_object.identity] = mapping.source_object.public()
                    else:
                        self.expression(routine, mapping.actual)
                nested = (routine.qualified, id(call)) in self.team_calls
                self.team_call_depth += int(nested)
                try:
                    self.procedure(resolved.procedure)
                except (_DescriptorFunctionRequired, _DescriptorTerminationRequired):
                    raise
                except DescriptorWindowUnsupported as error:
                    raise _DescriptorTerminationRequired(routine.qualified, call, str(error)) from error
                finally:
                    self.team_call_depth -= int(nested)
            elif node.kind == "boundary":
                originals = graph.source_nodes(identity)
                if len(originals) == 1 and (
                        _kind(originals[0]) == "Assignment_Stmt" and self.library_constructor(routine, originals[0])
                        or _kind(originals[0]) == "Call_Stmt" and self.library_call(routine, graph, identity, originals[0])):
                    self.operations += 1
                    continue
                if len(originals) == 1 and _kind(originals[0]) == "Call_Stmt" and self.owned_record_call(routine, originals[0]):
                    self.operations += 1
                    continue
                if self.storage_group(routine, graph, identity):
                    self.operations += 1
                    continue
                if len(originals) == 1 and _kind(originals[0]) == "Block_Construct":
                    self.joined_block(routine, graph, identity, originals[0])
                    self.operations += 1
                    continue
                if len(originals) == 1 and _kind(originals[0]) in {"Write_Stmt", "Print_Stmt"}:
                    from compiler.frontend.descriptor_output import (
                        DescriptorOutputUnsupported,
                        original_intrinsic_output,
                    )

                    try:
                        output = original_intrinsic_output(self, routine, originals[0])
                    except DescriptorOutputUnsupported as error:
                        raise DescriptorWindowUnsupported(str(error)) from error
                    self.operations += 1
                    self.native_outputs[routine.qualified, identity] = {
                        **output, "source_graph": graph.identity, "original_node_id": identity}
                    continue
                if len(originals) == 1 and _kind(originals[0]) in {"Exit_Stmt", "Cycle_Stmt"}:
                    self.loop_control(routine, graph, identity, originals[0])
                    continue
                if len(originals) == 1 and _kind(originals[0]) == "Return_Stmt":
                    original = originals[0]
                    if (len(self.active) < 2 or self.active[-1] != routine.qualified
                            or tuple(original.items) != (None,)
                            or (routine.qualified, id(original)) in self.joined_returns):
                        raise DescriptorWindowUnsupported("descriptor window RETURN requires a bare complete-callee exit outside original OpenMP")
                    if any("allocatable" in binding.attributes and name not in routine.arguments
                           for name, binding in routine.scope.bindings.items()):
                        raise DescriptorWindowUnsupported("descriptor window RETURN local allocation finalization requires separate authority")
                    self.resolution(routine, original)
                    self.operations += 1
                    self.native_returns[routine.qualified, identity] = {
                        "procedure": routine.qualified, "source_graph": graph.identity,
                        "original_node_id": identity, "original_statement": str(original),
                        "complete_original_callee_required": True, "all_following_original_nodes_scanned": True,
                        "descriptor_evidence_only": True, "entry_definition_authority": False,
                        "payload_access_authorized": False, "execution_authorized": False}
                    continue
                if len(originals) == 1 and self.scalar_metadata_call(routine, originals[0]):
                    continue
                if len(originals) == 1 and self.scalar_clock(routine, originals[0]):
                    self.operations += 1
                    continue
                candidates = [proof for proof in self.join_analysis._native_environments.values()
                              if isinstance(proof, NativeEnvironmentProof) and len(originals) == 1
                              and proof.procedure == routine.qualified
                              and proof._call is originals[0]]
                if len(candidates) != 1:
                    reason = ("descriptor window unknown selected storage effects: boundary"
                              + (": " + node.details["reason"] if node.details.get("reason") else ""))
                    if len(originals) == 1 and _kind(originals[0]) == "Call_Stmt":
                        if self.terminating_call(routine, originals[0]):
                            self.operations += 1
                            continue
                        raise _DescriptorTerminationRequired(routine.qualified, originals[0], reason)
                    raise DescriptorWindowUnsupported(reason)
                proof = candidates[0].validate(self.join_analysis, routine.qualified, originals[0])
                self.resolution(routine, originals[0])
                self.operations += 1
                self.environments[proof.identity] = {**proof.public(), "descriptor_evidence_only": True,
                                                     "execution_authorized": False}
            else:
                raise DescriptorWindowUnsupported("descriptor window unknown selected storage effects: " + node.kind
                                       + (": " + node.details["reason"] if node.details.get("reason") else ""))

    def record_closure_names(self):
        # Reuse the retained complete SAME inventory; functions do not become
        # public structured subroutine entries or acquire numerical authority.
        return {member["procedure"] for proof in self.record_calls.values()
                for member in proof._storage._record["closure"]}

    def owned_record_call(self, routine, call):
        from compiler.frontend.component_bindings import _typename
        from compiler.frontend.descriptor_record_calls import original_record_call
        from compiler.frontend.storage_record_regions import OwnedRecordRegionUnsupported

        scope = self.analysis.source_scope_for(call, routine.scope)
        candidates = tuple(self.analysis._candidates(scope, call.items[0]))
        if len(candidates) != 1:
            return False
        callee = self.analysis.numerical_helpers.get(candidates[0])
        if callee is None or not any(_typename(binding.dtype) is not None and binding.rank
                                    for name, binding in callee.scope.bindings.items() if name in callee.arguments):
            return False
        if len(self.active) >= self.analysis.depth_limit or len(set(self.checked) | self.record_closure_names() | set(self.active)) >= self.analysis.procedure_limit:
            raise CompilationError("descriptor window direct call closure budget exhausted")
        try:
            proof = original_record_call(self.join_analysis, routine, call, self.record_numeric_allocations)
        except OwnedRecordRegionUnsupported as error:
            raise DescriptorWindowUnsupported(str(error)) from error
        members = proof._storage._record["closure"]
        entry = proof._region._roles[1].qualified
        edges = {member["procedure"]: tuple(call["association"]["procedure"] for call in member["calls"])
                 for member in members}
        if entry not in edges or len(edges) != len(members):
            raise CompilationError("descriptor window requires complete unique original record body closure")
        closure = tuple(edges)
        reached = set(self.active) | set(self.checked) | self.record_closure_names() | set(closure)
        if len(reached) > self.analysis.procedure_limit:
            raise CompilationError("descriptor window composed original call closure budget exhausted")

        def depth(name, path):
            if name in path or name in self.active or name not in edges:
                raise CompilationError("descriptor window recursive or incomplete original record call closure")
            return 1 + max((depth(child, (*path, name)) for child in edges[name]), default=0)

        if len(self.active) + depth(entry, ()) > self.analysis.depth_limit:
            raise CompilationError("descriptor window composed original call depth budget exhausted")
        for name in closure:
            from compiler.frontend.source_functions import _original

            original = _original(self.analysis, name)
            self.scope_authority(original.scope)
            if name in self.analysis.routines:
                graph = self.analysis.structure(name)
                self.source_budget.reached(original, graph, (graph.root_id,))
                self.checked[original.qualified] = graph.identity
            # Source functions stay in the complete retained SAME closure,
            # whose own inventory bounds full original scans. No function is
            # retagged as a public structured subroutine to mint graph credit.
        self.record_calls[(routine.qualified, id(call))] = proof
        return True

    def procedure(self, name):
        if name in self.active:
            raise CompilationError("descriptor window recursive call closure is unavailable")
        if self.team_call_depth:
            routine = self.analysis._require_original(name)
            if any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
                   for node in _walk(routine.execution)):
                raise DescriptorWindowUnsupported("descriptor window nested original OpenMP closure requires separate authority")
        if name in self.checked:
            return
        if len(self.active) >= self.analysis.depth_limit or len(set(self.checked) | self.record_closure_names() | set(self.active)) >= self.analysis.procedure_limit:
            raise CompilationError("descriptor window direct call closure budget exhausted")
        routine = self.analysis._require_original(name)
        # Namespace authority is separate from executable references. The
        # complete original child is still visited below, including every true
        # declaration bound; USE/definition spellings and unrelated CONTAINS
        # bodies are not synthetic whole-procedure lookup units.
        self.scope_authority(routine.scope)
        proofs = [proof for proof in getattr(self.analysis, "_native_prefix_proofs", {}).values()
                  if isinstance(proof, NativePrefixProof) and proof.procedure == name]
        if proofs:
            if len(proofs) != 1:
                raise CompilationError("descriptor window requires unique registered native-prefix authority")
            proof = proofs[0].validate(self.analysis, name)
            for item in proof.public()["closure"]:
                self.prefix_scopes[item["procedure"]] = proof
        graph = self.analysis.structure(name)
        if not graph.available:
            raise CompilationError("descriptor window child bounded original graph is unavailable")
        self.source_budget.reached(routine, graph, (graph.root_id,))
        self.joined_groups(routine, graph, (graph.root_id,))
        self.declaration_resolution(routine)
        for binding in routine.scope.bindings.values():
            if binding.dtype.startswith("type("):
                self.fixed_object(binding)
            elif binding.dtype not in {"real", "integer", "logical"} or binding.attributes & {"pointer", "volatile", "asynchronous"}:
                raise DescriptorWindowUnsupported("descriptor window child specification effects require separate authority")
            states = [proof for proof in self.join_analysis._native_environment_states.values()
                      if isinstance(proof, NativeLocalStateProof) and proof.procedure == name
                      and proof.binding is binding]
            if states:
                if len(states) != 1:
                    raise CompilationError("descriptor window native environment state requires unique original authority")
                state = states[0].validate(self.join_analysis, name, binding)
                self.environment_states[state.identity] = {**state.public(), "descriptor_evidence_only": True,
                                                          "execution_authorized": False}
            else:
                for axis in binding.shape_nodes:
                    self.resolution(routine, _children(axis))
                    self.expression(routine, _children(axis))
        for argument in routine.arguments:
            binding = routine.scope.bindings.get(argument)
            if (binding is None or binding.attributes & _UNSAFE
                    and not self.optional_scalar(routine, binding) and not self.optional_output(routine, binding)):
                raise DescriptorWindowUnsupported("descriptor window child formal storage is unproved")
            if self.optional_output(routine, binding) and binding.rank:
                self.fixed_vector_bounds(routine, binding)
            if "allocatable" in binding.attributes and binding.intent not in {"in", "inout"}:
                raise DescriptorWindowUnsupported("descriptor window allocatable OUT/unspecified entry may replace storage")
        self.active.append(name)
        try:
            self.visit(routine, graph, (graph.root_id,))
        finally:
            self.active.pop()
        self.checked[name] = graph.identity


def _module_binding(analysis, routine, root, check):
    """Authenticate original visible allocation storage, never its contents."""
    parts = root.split("::")
    module = analysis.modules.get(parts[0]) if len(parts) == 2 and parts[0] != "argument" else None
    binding = module.bindings.get(parts[1]) if module is not None else None
    original = getattr(analysis, "_source_module_bindings", {}).get(root)
    fixed = (binding is not None and "allocatable" not in binding.attributes
             and len(binding.shape_nodes) == binding.rank
             and all(_kind(axis) == "Explicit_Shape_Spec" for axis in binding.shape_nodes))
    if (binding is None or binding.root != root or not ("allocatable" in binding.attributes or fixed)
            or original is None or binding is not original[0]
            or _hash(binding.public()) != sha256(original[1].encode()).hexdigest()
            or binding.attributes & (_UNSAFE | {"target", "parameter"})
            or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
            or not 1 <= binding.rank <= 4):
        raise CompilationError("descriptor window requires original numeric module allocations or fixed arrays: " + root)
    specification = next((part for part in _children(module.node)
                          if _kind(part) == "Specification_Part"), None)
    if any(_kind(node) in {"Common_Stmt", "Equivalence_Stmt", "Codimension_Attr_Spec"}
           or _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
           for node in _walk(specification)):
        raise CompilationError("descriptor window module association or OpenMP storage requires separate authority")
    if any(_kind(node) == "Entity_Decl" and str(node.items[0]).lower() == binding.name
           and node.items[2] is not None for node in _walk(specification)):
        raise CompilationError("descriptor window coarray module allocation requires separate authority")
    names, visited = set(), set()

    def collect(scope):
        if scope is None or id(scope) in visited:
            return
        check.tick()
        check.scope_authority(scope)
        visited.add(id(scope))
        for name in (*scope.bindings, *scope.imports):
            check.tick()
            names.add(name)
        for imported in scope.wildcards:
            collect(analysis.modules.get(imported))
        for imported, _remote in scope.imports.values():
            collect(analysis.modules.get(imported))
        collect(scope.parent)

    collect(routine.scope)
    for name in sorted(names):
        check.tick()
        if analysis._binding(routine.scope, name) is binding:
            # Even an otherwise unused requested root must authenticate the
            # exact original import/rename/reexport visibility used to bind it.
            # This is one registration unit, not a synthetic source statement.
            check.resolution(routine, (), names=(name,))
            return binding, name, module
    raise CompilationError("descriptor window module allocation is not originally visible: " + root)


def _fixed_module_bounds(analysis, module, binding, check):
    """Exact static numeric declaration; no PARAMETER or payload authority."""
    check.scope_authority(module)
    check.object_builder.scope(module)
    specification = next((part for part in _children(module.node)
                          if _kind(part) == "Specification_Part"), None)
    declaration = None
    consumed_axes = None
    for node in _walk(specification):
        check.tick()
        if _kind(node) in {"Bind_Stmt", "Cray_Pointer_Stmt", "Namelist_Stmt"}:
            raise CompilationError("fixed module storage has unproved declaration association")
        if _kind(node) != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = node.items
        dimension = next((item.items[1] for item in _children(attributes)
                          if _kind(item) == "Dimension_Attr_Spec"), None)
        for entity in _children(entities):
            if str(entity.items[0]).lower() != binding.name:
                continue
            if declaration is not None or entity.items[2] is not None:
                raise CompilationError("fixed module declaration is duplicated or coindexed")
            axes = tuple(_children(entity.items[1] if entity.items[1] is not None else dimension))
            if (len(axes) != binding.rank or any(a is not b for a, b in zip(axes, binding.shape_nodes, strict=True))
                    or any(_kind(axis) != "Explicit_Shape_Spec" or axis.items[1] is None for axis in axes)
                    or _kind(dtype) != "Intrinsic_Type_Spec"):
                raise CompilationError("fixed module descriptor differs from its original declaration")
            declaration = node
            consumed_axes = axes
    if declaration is None:
        raise CompilationError("fixed module original declaration is unavailable")
    consumed_declaration = _canonical(binding.public())

    def scope_authority(scope):
        check.scope_authority(scope)
        check.object_builder.scope(scope)
        check.fixed_module_scopes[id(scope)] = scope

    check.resolution_units += 1
    _validate_resolution(analysis, module, _walk(declaration), tick=check.resolution_tick,
                         _scope_validator=scope_authority)
    scope_authority(module)
    dtype = declaration.items[0]
    base, selector = dtype.items
    base = str(base).lower()
    kind = 8 if base == "double precision" else 4
    if base == "double precision":
        base = "real"
    if selector is not None:
        from compiler.ir import SourceLocation
        kind = module.kinds.integer(selector.items[1], SourceLocation(str(module.path)))
    if (base, kind) != (binding.dtype, binding.kind):
        raise CompilationError("fixed module type differs from its original declaration")
    if (module.bindings.get(binding.name) is not binding or binding.declaring_scope is not module
            or _canonical(binding.public()) != consumed_declaration
            or len(binding.shape_nodes) != len(consumed_axes)
            or any(a is not b for a, b in zip(binding.shape_nodes, consumed_axes, strict=True))):
        raise CompilationError("fixed module consumed declaration changed during resolution")
    try:
        bounds = check.object_builder.bounds(module, binding)
    except CompilationError as error:
        raise CompilationError("fixed module storage requires original constant INTEGER bounds") from error
    check.fixed_module_bindings[id(binding)] = (
        binding, module, consumed_axes, consumed_declaration)
    return [list(axis) for axis in bounds]


def _derive(analysis, procedure, identities, roots, module_roots=(), *, _defer_publication=False,
            _discover_functions=False):
    if (not isinstance(identities, tuple) or not identities
            or any(not isinstance(identity, str) for identity in identities)):
        raise CompilationError("descriptor window requires exact original structured node IDs")
    # Selection already authenticates the exact original routine and graph.
    graph, selected = analysis._selected_source(procedure, identities)
    routine = analysis.routines[procedure]
    # The selection is a complete original graph interval, not a numerical
    # optimizer search window. Source membership and the charged traversal
    # bound its size; no statement is omitted to satisfy operation_limit.
    if len(identities) > len(graph.nodes):
        raise CompilationError("descriptor window selection exceeds original graph inventory")
    check = _Check(analysis)
    if _discover_functions:
        check.function_demands = {}
    check.scope_authority(routine.scope)
    sequences = []
    for parent in graph.nodes.values():
        if parent.kind == "sequence" and selected[0] in parent.children:
            first = parent.children.index(selected[0])
            if parent.children[first:first + len(selected)] == selected:
                sequences.append(parent.id)
    headers = [node.id for node in graph.nodes.values() if node.kind == "loop"
               and len(selected) == 1 and selected[0] in node.children
               and graph.nodes[selected[0]].details.get("evaluation") == "loop_header"]
    conditions = [node.id for node in graph.nodes.values() if node.kind == "branch"
                  and len(selected) == 1 and any(condition == selected[0] for condition, _body in node.alternatives)
                  and graph.nodes[selected[0]].details.get("evaluation") == "condition"]
    if len(sequences) != 1 and len(headers) != 1 and len(conditions) != 1:
        raise CompilationError("descriptor window requires one contiguous original immediate sequence")
    if (not isinstance(roots, tuple) or not isinstance(module_roots, tuple)
            or not (roots or module_roots)
            or len(roots) > len(routine.scope.bindings)
            or len(module_roots) > len(getattr(analysis, "_source_module_bindings", {}))
            or any(not isinstance(root, str) for root in (*roots, *module_roots))
            or len(set((*roots, *module_roots))) != len(roots) + len(module_roots)):
        raise CompilationError("descriptor window requires bounded unique original formal roots")
    check.source_budget.reached(routine, graph, selected)
    bindings = {binding.root: binding for binding in routine.scope.bindings.values()}
    states, numeric_record_bindings = [], []
    for root in roots:
        binding = bindings.get(root)
        ordinary = (binding is not None and "allocatable" not in binding.attributes
                    and binding.name in routine.arguments
                    and binding.declaring_scope is routine.scope
                    and not binding.attributes & (_UNSAFE | {"target", "parameter", "save"})
                    and len(binding.shape_nodes) == binding.rank
                    and all(_kind(axis) in {"Assumed_Shape_Spec", "Explicit_Shape_Spec"}
                            for axis in binding.shape_nodes))
        if (binding is None or not (ordinary or "allocatable" in binding.attributes) or binding.attributes & _UNSAFE
                or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
                or not 1 <= binding.rank <= 4):
            raise CompilationError("descriptor window requires original numeric allocation or ordinary array formal: " + root)
        numeric_record_bindings.append(binding)
        local = (original_local_allocation(analysis, routine, binding)
                 if binding.name not in routine.arguments else None)
        state = {"resource": root, "resource_kind": ("original_local_save_allocatable" if local is not None
                                                    else "original_ordinary_array_formal" if ordinary
                                                    else "original_allocatable_formal"),
                       "original_binding": binding.public(),
                       "stable": True, "original_descriptor_required": True,
                       "descriptor_changes_within_selection": False, "escapes_within_selection": False,
                       "requires_prior_noescape_and_current_allocation_proof": True,
                       "requires_original_allocation_presence_and_alias_guards": True,
                       "execution_requires_runtime_guard": True,
                       "entry_definition_authority": False}
        if local is not None:
            state.update(local_storage_identity=local.identity, requires_original_runtime_bounds=True)
        if ordinary:
            # An ordinary nonpointer/nonallocatable dummy cannot reassociate
            # its own descriptor during this activation. The unchanged full
            # selected-closure scanner below still rejects mutations, escapes
            # and unknown calls. This proves neither ownership of the actual
            # allocation nor any payload, alias or caller-lifetime premise.
            state.update(borrowed_descriptor=True,
                descriptor_scope="exact original procedure activation and checked selected closure",
                caller_storage_lifetime_authority=False, physical_allocation_origin_authority=False,
                payload_access_authorized=False, alias_authority=False)
        states.append(state)
    for root in module_roots:
        binding, spelling, module = _module_binding(analysis, routine, root, check)
        numeric_record_bindings.append(binding)
        if "allocatable" not in binding.attributes:
            states.append({"resource": root, "resource_kind": "original_module_fixed_array",
                "original_binding": binding.public(), "original_visible_spelling": spelling,
                "declaration_source": str(module.path),
                "declaration_source_sha256": analysis.sources[str(module.path)],
                "fixed_bounds": _fixed_module_bounds(analysis, module, binding, check),
                "stable": True, "original_descriptor_required": True,
                "descriptor_changes_within_selection": False, "escapes_within_selection": False,
                "storage_lifetime": "original fixed module object within checked selected closure",
                "requires_prior_noescape_proof": True, "requires_original_alias_guards": True,
                "execution_requires_runtime_guard": True, "entry_definition_authority": False,
                "payload_access_authorized": False, "alias_authority": False})
            continue
        states.append({"resource": root, "resource_kind": "original_module_allocatable",
                       "original_binding": binding.public(), "original_visible_spelling": spelling,
                       "declaration_source": str(module.path),
                       "declaration_source_sha256": analysis.sources[str(module.path)],
                       "stable": True, "original_descriptor_required": True,
                       "descriptor_changes_within_selection": False, "escapes_within_selection": False,
                       "requires_prior_noescape_and_current_allocation_proof": True,
                       "requires_original_allocation_presence_and_alias_guards": True,
                       "requires_original_runtime_bounds": True,
                       "execution_requires_runtime_guard": True, "entry_definition_authority": False})
    check.protected_allocations = frozenset((*roots, *module_roots))
    check.record_numeric_allocations = tuple(numeric_record_bindings)
    check.active.append(procedure)
    try:
        check.joined_groups(routine, graph, selected)
        check.visit(routine, graph, selected)
    except DescriptorWindowUnsupported as error:
        # A soft semantic decline must not hide source/namespace mutation
        # after a cached scope check. Hard failures and budget exits stay hard.
        check.publish(procedure, graph.authority_identity)
        if check.function_demands:
            raise _DescriptorFunctionsRequired(check.function_demands.values()) from error
        raise
    finally:
        check.active.pop()
    if _discover_functions:
        check.publish(procedure, graph.authority_identity)
        # Discovery can never issue a window, even if a callback clears its
        # private collector. Only a later normal scan may publish authority.
        raise _DescriptorFunctionsRequired((check.function_demands or {}).values())
    attachments = _PublicationAttachments(analysis, check)
    record = {"schema_version": DESCRIPTOR_WINDOW_VERSION, "procedure": procedure,
            "source_identity": analysis.inputs.identity(), "structured_identity": graph.identity,
            "analysis_identity": graph.authority_identity, "selected_node_ids": list(selected),
            "parent_sequence": sequences[0] if sequences else None,
            "parent_node": sequences[0] if sequences else headers[0] if headers else conditions[0],
            "selection_role": ("original contiguous sequence" if sequences else
                               "original counted header" if headers else "original condition"),
            "original_guard": list(graph.nodes[selected[0]].guard),
            "resources": states, "module_roots": list(module_roots), "source_call_closure": check.checked,
            "native_source_function_requirements": list(check.functions.values()),
            "native_predicate_requirements": list(check.predicates.values()),
            "native_prefix_descriptor_requirements": list(check.prefixes.values()),
            "native_joined_descriptor_requirements": list(check.joins.values()),
            "native_environment_descriptor_requirements": list(check.environments.values()),
            "native_environment_state_requirements": list(check.environment_states.values()),
            "fixed_source_object_requirements": list(check.objects.values()),
            "source_object_mappings": list(check.object_mappings.values()),
            "source_operations": check.operations, "source_traversal_entries": check.nodes,
            "resolution_validation": {"mode": "original_semantic_units", "units": check.resolution_units,
                                      "work_entries": check.resolution_work,
                                      "syntax_entries": check.resolution_syntax,
                                      "lookup_transitions": check.resolution_lookups,
                                      "per_unit_lookup_limit": analysis.operation_limit,
                                      "depth_limit": analysis.depth_limit,
                                      "procedure_limit": analysis.procedure_limit,
                                      "aggregate_source_scan_limit": check.source_budget.limit},
            "prefix_effects_proved": False, "entry_definition_authority": False,
            "reachability_authority": False, "execution_authorized": False, "gpu_legality_established": False,
            "requires_close_before_excluded_tail": True, "freshness_by_address_or_shape": False,
            "limits": ["explicit original numeric formals/module allocations only",
                       "module allocation authority is conditional and exact to this selection and checked calls",
                       "no deferred groups or whole derived-object effects",
                       "fixed object mappings prove original storage/type only, never native continuation or numeric ABI",
                       "OpenMP requires registered whole joined-group or native-prefix proof and full descriptor rescan",
                       "excluded prefix and tail have no lifetime or definition authority",
                       "registered readonly source-function proof required before query",
                       "selected closure replacement, escapes, unknown calls and effects reject"]}
    if check.source_budget.limit > 65536:
        record["schema_version"] = DESCRIPTOR_WINDOW_SOURCE_ACCOUNTING_VERSION
        record["source_scan_accounting"] = check.source_budget.public()
    if check.metadata_objects:
        record["native_metadata_descriptor_requirements"] = list(check.metadata_objects.values())
    if check.descriptor_functions:
        record["native_descriptor_source_function_requirements"] = list(check.descriptor_functions.values())
    if check.intrinsics:
        record["native_intrinsic_descriptor_requirements"] = list(check.intrinsics.values())
    if check.metadata_calls:
        record["native_metadata_call_descriptor_requirements"] = list(check.metadata_calls.values())
    if check.native_returns:
        record["native_return_descriptor_requirements"] = list(check.native_returns.values())
    if check.native_loops:
        record["native_loop_descriptor_requirements"] = list(check.native_loops.values())
    if check.native_loop_controls:
        record["native_loop_control_descriptor_requirements"] = list(check.native_loop_controls.values())
    if check.optional_scalars:
        record["native_optional_scalar_requirements"] = list(check.optional_scalars.values())
    if check.optional_arrays:
        record["native_optional_array_requirements"] = list(check.optional_arrays.values())
    if check.terminating_calls:
        record["native_terminating_descriptor_requirements"] = list(check.terminating_calls.values())
    if check.native_outputs:
        record["schema_version"] = DESCRIPTOR_WINDOW_INTRINSIC_OUTPUT_VERSION
        record["native_intrinsic_output_requirements"] = list(check.native_outputs.values())
    if check.storage_functions:
        record["schema_version"] = DESCRIPTOR_WINDOW_STORAGE_FUNCTION_VERSION
        record["native_storage_function_requirements"] = list(check.storage_functions.values())
        record["limits"] = [item for item in record["limits"]
                            if item != "registered readonly source-function proof required before query"]
        record["limits"].append("source storage projection grants descriptor effects only; native access and reached retirement remain independent")
    if any(state["resource_kind"] == "original_ordinary_array_formal" for state in states):
        record["schema_version"] = DESCRIPTOR_WINDOW_BORROWED_ARRAY_VERSION
        record["limits"].append("ordinary dummy descriptor stability grants no caller storage lifetime, allocation origin, aliases or initialized payload")
    if any(state["resource_kind"] == "original_module_fixed_array" for state in states):
        record["schema_version"] = DESCRIPTOR_WINDOW_FIXED_MODULE_VERSION
        record["limits"].append("fixed module storage retains original mutable payload and independent native effects, runtime aliases and entry definitions")
    if check.scalar_clocks:
        record["schema_version"] = DESCRIPTOR_WINDOW_SCALAR_CLOCK_VERSION
        record["native_scalar_clock_descriptor_requirements"] = list(check.scalar_clocks.values())
        record["limits"].append("original scalar clocks retain native execution/completion effects; descriptor evidence grants no payload or execution authority")
    if check.storage_groups:
        from compiler.frontend.descriptor_storage_groups import DESCRIPTOR_WINDOW_STORAGE_GROUP_VERSION
        record["schema_version"] = DESCRIPTOR_WINDOW_STORAGE_GROUP_VERSION
        record["native_storage_group_requirements"] = list(check.storage_groups.values())
        record["limits"].append("complete original deferred groups use independent team storage and mutation proofs; no numerical completion, payload access or execution authority")
    if check.library_states or check.library_calls:
        from compiler.frontend.library_storage_calls import DESCRIPTOR_WINDOW_LIBRARY_STORAGE_VERSION
        record["schema_version"] = DESCRIPTOR_WINDOW_LIBRARY_STORAGE_VERSION
        record["native_private_library_state_requirements"] = [proof._declaration.public()
            for proof in check.library_states.values()]
        record["native_library_storage_requirements"] = [storage.public()
            for _proof, storage in check.library_calls.values()]
        record["limits"].append("explicit pinned library storage facts grant descriptor preservation and noescape only; caller aliases, payload access, definitions and execution remain independent")
    if check.fixed_associations:
        from compiler.frontend.descriptor_associations import DESCRIPTOR_WINDOW_ASSOCIATION_VERSION
        record["schema_version"] = DESCRIPTOR_WINDOW_ASSOCIATION_VERSION
        record["native_fixed_scalar_associations"] = [proof.public() for proof in check.fixed_associations.values()]
        record["limits"].append("fixed scalar lexical selectors retain a complete original body scan; no whole-object access, alias separation or execution authority")
    def retained_scalar_team(storage):
        return storage.get("version") == 2 or (storage.get("version") == 4 and (
            storage.get("retained_single_regions", 0) or storage.get("intrinsic_scalar_reductions")))

    if any(not group.available or retained_scalar_team(proof.public()["team_storage"])
           or proof.public()["storage_mutation"].get("fixed_scalar_blocks")
           for _routine, _graph, group, proof in check.storage_group_tokens.values()) or any(
               retained_scalar_team(item.get("storage_structure", {})) for item in check.joins.values()):
        from compiler.frontend.descriptor_blocks import DESCRIPTOR_WINDOW_BLOCK_VERSION

        record["schema_version"] = DESCRIPTOR_WINDOW_BLOCK_VERSION
        record["limits"].append("complete source joined storage is independent of numerical unit inventory; scalar reduction, SINGLE and fixed BLOCK declarations grant no execution or numerical completion")
    if check.numeric_component_reads:
        record["schema_version"] = DESCRIPTOR_WINDOW_COMPONENT_READ_VERSION
        record["native_numeric_component_reads"] = [
            {"procedure": routine.qualified, **json.loads(public)}
            for routine, _node, _root, _indices, public, _scopes, _syntax in check.numeric_component_reads.values()]
        record["limits"].append("unchanged original numeric component reads establish descriptor effects only; allocation, definitions, payload layout and execution remain independent")
    if any(item["intrinsic"] in {"matmul", "transpose"} for item in check.intrinsics.values()):
        record["schema_version"] = DESCRIPTOR_WINDOW_MATRIX_INTRINSIC_VERSION
        record["limits"].append("typed native matrix intrinsic reads retain original execution and shape checks; no payload evaluation, numerical lowering or execution authority")
    if (any(proof._team.public().get("retained_master_regions", 0)
            for *_group, proof in check.storage_group_tokens.values())
            or any(item.get("storage_structure", {}).get("retained_master_regions", 0)
                   for item in check.joins.values())
            or any(proof._team.public().get("retained_master_regions", 0)
                   for proof, _baseline in check.joined_blocks.values())):
        from compiler.frontend.descriptor_teams import DESCRIPTOR_WINDOW_MASTER_VERSION

        record["schema_version"] = DESCRIPTOR_WINDOW_MASTER_VERSION
        record["limits"].append("original MASTER bodies retain the parent parallel join; END MASTER implies no barrier or execution completion")
    if check.joined_blocks:
        from compiler.frontend.descriptor_block_storage import DESCRIPTOR_WINDOW_JOINED_BLOCK_VERSION

        record["schema_version"] = DESCRIPTOR_WINDOW_JOINED_BLOCK_VERSION
        record["native_joined_block_requirements"] = [proof.public() for proof, _baseline in check.joined_blocks.values()]
        record["limits"].append("ordinary BLOCK boundaries require exact complete joined-team declarations and full source storage closure, independent of numerical group size; no execution, access or definition authority")
    if check.private_allocations:
        record["schema_version"] = DESCRIPTOR_WINDOW_PRIVATE_ALLOCATION_VERSION
        record["native_private_allocation_requirements"] = [
            proof.public() for proof, _baseline in check.private_allocations.values()]
        record["requires_unregistered_private_allocations"] = sorted({
            proof.binding.root for proof, _baseline in check.private_allocations.values()})
        record["limits"].append("original private allocation mutations remain native; every changed private allocation must be absent from the complete live domain registration inventory, including host reservations; no payload, initialization or allocation-presence authority")
    if (any(proof._team.public().get("implicit_worksharing_endings")
            for *_group, proof in check.storage_group_tokens.values())
            or any(item.get("storage_structure", {}).get("implicit_worksharing_endings")
                   for item in check.joins.values())
            or any(proof._team.public().get("implicit_worksharing_endings")
                   for proof, _baseline in check.joined_blocks.values())
            or any(proof.public().get("implicit_worksharing_endings") for proof in check.storage_teams)):
        from compiler.frontend.descriptor_teams import DESCRIPTOR_WINDOW_IMPLICIT_WORKSHARING_VERSION

        record["schema_version"] = DESCRIPTOR_WINDOW_IMPLICIT_WORKSHARING_VERSION
        record["limits"].append("omitted original worksharing END DO retains the associated counted-loop completion and default barrier inside the complete parent team; no internal cut, payload, definition or execution authority")
    if check.worksharing_storage:
        from compiler.frontend.descriptor_worksharing import DESCRIPTOR_WINDOW_ORIGINAL_WORKSHARING_VERSION

        record["schema_version"] = DESCRIPTOR_WINDOW_ORIGINAL_WORKSHARING_VERSION
        record["native_worksharing_storage_requirements"] = [proof.public()
            for _name, (proof, _baseline) in sorted(check.worksharing_storage.items())]
        record["limits"].append("original orphan worksharing retains the whole unchanged native callee; no internal cut, completion, payload, definition or execution authority")
    if any(item["intrinsic"] == "scalar_array_assignment" for item in check.intrinsics.values()):
        from compiler.frontend.descriptor_scalar_fill import DESCRIPTOR_WINDOW_SCALAR_FILL_VERSION

        record["schema_version"] = DESCRIPTOR_WINDOW_SCALAR_FILL_VERSION
        record["limits"].append("primitive scalar broadcast requires the original allocated array and retains its descriptor; no allocation presence, initializedness, payload access or execution authority")
    if check.record_calls:
        from compiler.frontend.descriptor_record_calls import DESCRIPTOR_WINDOW_RECORD_REGION_VERSION
        record.update(schema_version=DESCRIPTOR_WINDOW_RECORD_REGION_VERSION,
                      owned_record_call_requirements=[proof.public() for proof in check.record_calls.values()])
    if any(projection.get("readonly_record_associations") for projection in check.storage_functions.values()):
        record["schema_version"] = DESCRIPTOR_WINDOW_READONLY_RECORD_VERSION
        record["limits"].append("readonly record source-function associations classify storage effects only; no object payload, allocation presence, definition or execution authority")
    if _defer_publication:
        return record, check, attachments
    _publish_derivation(analysis, procedure, graph.authority_identity, record, check)
    _finish_derivation(analysis, attachments)
    return record


@dataclass(frozen=True)
class DescriptorWindowProof:
    procedure: str
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _selected: tuple[str, ...] = field(repr=False, compare=False)
    _roots: tuple[str, ...] = field(repr=False, compare=False)
    _module_roots: tuple[str, ...] = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _worksharing: tuple = field(default=(), repr=False, compare=False)
    _source_fact: object = field(default=None, repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, selected_node_ids):
        _proof_registry_items(getattr(analysis, "_descriptor_window_proofs", {}))
        if (analysis is not self._analysis or procedure != self.procedure
                or not isinstance(selected_node_ids, tuple) or selected_node_ids != self._selected
                or getattr(analysis, "_descriptor_window_proofs", {}).get(self.identity) is not self):
            raise CompilationError("descriptor window lacks registered original interval authority")
        if self._source_fact is not None:
            return self._source_fact.require(self)
        if getattr(analysis, "_source_session", None) is not None:
            raise CompilationError("mutable diagnostic proof cannot enter a sealed authority context")
        worksharing = self._worksharing
        if any(proof._analysis is not analysis
                or analysis.routines.get(proof._routine.qualified) is not proof._routine for proof in worksharing):
            raise CompilationError("descriptor window worksharing belongs to another original analysis")
        baselines = tuple((proof, proof.checkpoint()) for proof in worksharing)
        if _freeze(_derive(analysis, procedure, self._selected, self._roots, self._module_roots)) != self._record:
            raise CompilationError("descriptor window original source authority changed")
        if self._worksharing is not worksharing:
            raise CompilationError("descriptor window worksharing attachment changed")
        expected = [proof.public() for proof in worksharing]
        if _thaw(self._record.get("native_worksharing_storage_requirements", ())) != expected:
            raise CompilationError("descriptor window lost a retained worksharing requirement")
        for proof, baseline in baselines:
            proof.require_checkpoint(baseline)
        return self

    @property
    def module_roots(self):
        return self._module_roots

    def resource_state(self, root):
        self.validate(self._analysis, self.procedure, self._selected)
        states = {state["resource"]: state for state in self._record["resources"]}
        if root not in states:
            raise CompilationError("descriptor window resource was not proved")
        return _thaw(states[root])


def prove_descriptor_window(analysis, procedure, selected_node_ids, requested_roots=(), *, module_roots=()):
    """Register selected-span stability, never prefix freshness or definitions."""
    from compiler.frontend.source_facts import DESCRIPTOR, dependencies, issue, lookup

    query = procedure, selected_node_ids, requested_roots, module_roots
    sealed = getattr(analysis, "_source_session", None) is not None
    context = analysis.source_authority_context() if sealed else None
    if all(type(value) is tuple and all(type(item) is str for item in value)
           for value in (selected_node_ids, requested_roots, module_roots)):
        previous = lookup(analysis, DESCRIPTOR, query)
        if previous is not None:
            return previous.validate(analysis, procedure, selected_node_ids)
    factory = _FactoryAttachments(analysis, procedure)
    requested = set()
    discover_functions = False

    def issue_function(demand):
        factory.require()
        key = demand.procedure, id(demand.expression)
        if key in requested:
            raise CompilationError("descriptor source function admission made no progress") from demand
        if len(requested) >= analysis.procedure_limit:
            raise CompilationError("descriptor source function admission request budget exhausted") from demand
        requested.add(key)
        # Preserve already-qualified readonly tokens/records where their
        # issuer admits the original call. A separate storage projection
        # does not need readonly payload semantics or configured PURE.
        try:
            legacy = (analysis.descriptor_source_function_call(demand.procedure, demand.expression)
                      if demand.legacy_eligible else None)
        except CompilationError:
            legacy = None
        if legacy is None:
            projection = analysis.storage_mutation_effects(demand.procedure, expression=demand.expression)
            if not projection.preserves_descriptors():
                raise DescriptorWindowUnsupported("descriptor window source-function storage mutation/escape closure is unavailable") from demand
        factory.require()

    while True:
        try:
            record, check, attachments = _derive(analysis, procedure, selected_node_ids, requested_roots,
                                                  module_roots, _defer_publication=True,
                                                  _discover_functions=discover_functions)
            break
        except _DescriptorFunctionsRequired as batch:
            for demand in batch.demands:
                issue_function(demand)
            discover_functions = False
        except _DescriptorFunctionRequired as demand:
            issue_function(demand)
            discover_functions = True
        except _DescriptorTerminationRequired as demand:
            discover_functions = False
            factory.require()
            from compiler.frontend.descriptor_termination import (
                DescriptorTerminationUnsupported,
                prove_descriptor_termination,
            )

            key = "termination", demand.procedure, id(demand.call)
            if key in requested:
                raise CompilationError("descriptor terminating-call admission made no progress") from demand
            if len(requested) >= analysis.procedure_limit:
                raise CompilationError("descriptor terminating-call admission request budget exhausted") from demand
            requested.add(key)
            try:
                prove_descriptor_termination(analysis, demand.procedure, demand.call)
            except DescriptorTerminationUnsupported as error:
                raise DescriptorWindowUnsupported(str(demand)) from error
    factory.require()
    _publish_derivation(analysis, procedure, record["analysis_identity"], record, check)
    identity, frozen = _hash(record), _freeze(record)
    registry = factory.registry
    _proof_registry_items(registry)
    proof = dict.get(registry, identity)
    if identity in registry:
        if (type(proof) is not DescriptorWindowProof or proof._analysis is not analysis
                or proof.procedure != procedure or proof._selected != selected_node_ids
                or proof._roots != requested_roots or proof._module_roots != module_roots
                or proof.identity != identity or proof._record != frozen
                or tuple(item.public() for item in proof._worksharing) != tuple(item.public()
                    for _name, (item, _baseline) in sorted(check.worksharing_storage.items()))):
            raise CompilationError("descriptor window cached original interval authority changed")
    else:
        from compiler.frontend.storage_mutation_effects import source_registry_budget

        if len(registry) >= source_registry_budget(analysis):
            raise CompilationError("descriptor window proof registry budget exhausted")
    _finish_derivation(analysis, attachments)
    factory.require()
    if identity not in registry:
        # Construction and insertion are after the final callback. No fresh
        # token can be exposed to a verifier before its registered publication.
        proof = DescriptorWindowProof(procedure, identity, analysis, selected_node_ids, requested_roots,
                                      module_roots, frozen, tuple(proof for _name, (proof, _baseline)
                                          in sorted(check.worksharing_storage.items())))
        registry[identity] = proof
    if context is not None:
        context.require(analysis)
        retained = dependencies(attachments.registrations)
        proof = issue(analysis, DESCRIPTOR, query, proof, retained)
    return proof
