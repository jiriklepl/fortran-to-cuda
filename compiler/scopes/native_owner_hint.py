"""A bounded CPU performance hint for a fresh original owning invocation.

The hint selects unchanged original CPU semantics. It does not authorize an
element read, definition, GPU operation, estimate, or borrowed coherent access.
The existing checked execution plan supplies only its static policy identity;
entry observations contain descriptors, never array or floating-point values.
"""

import json
import re
from dataclasses import dataclass
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.source import _name

VERSION = 1
MAX_WORDS = 480  # Runtime quiet-misses if source plus environment exceeds 512.


def latch(member):
    """An original lexical descendant may host-associate its root's local.

    A separately compiled module companion cannot reference its caller's
    latch. Hits call that companion's unchanged ordinary entry instead.
    """
    root = member.owner
    if (root.builder.config.scope_execution != "graph" or root.builder.config.policy != "auto"
            or any(hasattr(root, name) for name in ("closed_selection", "interval_site", "window_proof"))):
        return None
    current = member
    while current is not root:
        if current.control_guard is not None:
            return None
        current = current.parent
        if current is None:
            return None
    return _name("fort_native_invocation_", root.routine.qualified)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Descriptor:
    binding: object
    name: str


@dataclass(frozen=True)
class NativeOwnerHint:
    owner: object
    plan: object
    identity: str
    static: tuple[str, ...]
    descriptors: tuple[Descriptor, ...]
    word_count: int

    def validate(self):
        self.plan.validate()
        if self.owner.execution_plan is not self.plan or self.owner.parent is not None:
            raise CompilationError("native invocation hint lost its original owning execution plan")
        for descriptor in self.descriptors:
            current = self.owner.builder.visible_binding(self.owner.routine, descriptor.binding.root)
            if current is not descriptor.binding:
                raise CompilationError("native invocation hint changed its observed descriptor binding")
        return self

    def public(self):
        return {"schema_version": VERSION, "eligible": True, "identity": self.identity,
                "execution_identity": self.plan.execution_identity,
                "identity_includes": ["source inputs", "numerical configuration", "reviewed entry facts",
                                      "profile", "entry epoch and body-effect policies"],
                "entry_observations": [{"resource": item.binding.root, "name": item.name,
                                        "rank": item.binding.rank,
                                        "allocation_guard": "allocatable" in item.binding.attributes,
                                        "values": "descriptor bounds, extents and contiguity only"}
                                       for item in self.descriptors],
                "metadata_words": self.word_count,
                "selection": "conservative native performance hint; not a cost or definition proof",
                "hit": "immutable invocation latch; original CPU operations once; no managed owner",
                "miss": "unchanged current source, definition, profile and GPU validation",
                "population": "successful owning normal end; no GPU selection, attempt or acceptance; no unexpected error",
                "expected_prework_native_fallback": "eligible; no claim that every rejected query was priced",
                "quality_limit": "bounded native hits may suppress a newly profitable path until forced exploration",
                "estimate_available": False, "borrowed_coherent_owner": "never eligible"}

    def emission(self):
        """Return root USE/specification, reached entry, and normal-end code."""
        self.validate()
        owner, kind = self.owner, self.owner.kind
        native = latch(owner)
        obs = _name("fort_native_observation_", owner.routine.qualified)
        words = _name("fort_native_words_", owner.routine.qualified)
        observed = _name("fort_native_observed_", owner.routine.qualified)
        status = _name("fort_native_status_", owner.routine.qualified)
        hit = _name("fort_native_hit_", owner.routine.qualified)
        imports = ["use fort_scoped_memory, only: fort_scope_native_hint_observation_v1, &",
                   "fort_scope_native_hint_begin_v1, fort_scope_native_hint_finish_v1"]
        specification = [f"type(fort_scope_native_hint_observation_v1) :: {obs}",
                         f"integer({kind}) :: {words}({self.word_count})", f"logical :: {observed}"]
        # Locals used only by the optional hint live in a short entry block.
        entry = [f"{observed} = .false.", f"if ({owner.enabled}) then", "block",
                 "use iso_c_binding, only: c_int, c_size_t", f"integer(c_int) :: {status}, {hit}"]
        index = 1
        for digest in self.static:
            for offset in range(0, len(digest), 16):
                entry += [f"{words}({index}) = int(z'{digest[offset:offset+16]}',kind={kind})"]
                index += 1
        for descriptor in self.descriptors:
            binding, name = descriptor.binding, descriptor.name
            length = 2 + 2*binding.rank
            entry += [f"{words}({index}:{index+length-1}) = 0_{kind}"]
            if "allocatable" in binding.attributes:
                entry += [f"if (allocated({name})) then"]
            entry += [f"{words}({index}) = 1_{kind}"]
            index += 1
            entry += [f"if (is_contiguous({name})) {words}({index}) = 1_{kind}"]
            index += 1
            for axis in range(1, binding.rank+1):
                entry += [f"{words}({index}) = lbound({name},{axis},kind={kind})",
                          f"{words}({index+1}) = size({name},{axis},kind={kind})"]
                index += 2
            if "allocatable" in binding.attributes:
                entry += ["endif"]
        if index != self.word_count+1:
            raise CompilationError("native invocation hint metadata inventory changed")
        entry += [f"{hit} = 0", f"{status} = fort_scope_native_hint_begin_v1( &",
                  f"{words}, {self.word_count}_c_size_t, {obs}, {hit})",
                  f"if ({status} == 0_c_int) then", f"{observed} = {hit} == 0_c_int",
                  f"{native} = {hit} /= 0_c_int", "endif", "end block", "endif",
                  f"if ({native}) {owner.enabled} = .false."]
        ending = [f"if ({observed}) then", "block", "use iso_c_binding, only: c_int",
                  f"integer(c_int) :: {status}", f"{status} = fort_scope_native_hint_finish_v1({obs})",
                  "end block", "endif"]
        return imports, specification, entry, ending


def prepare(owner):
    """Issue only for a complete ordinary graph AUTO invocation.

    Optional/pointer descriptors are omitted, not speculatively tested. A CPU
    choice remains semantically valid when unobserved path/value inputs change;
    bounded forced exploration is the explicit placement quality limit.
    """
    if latch(owner) is None or owner.parent is not None or owner.builder.config.profile is None:
        return None, "not a complete graph AUTO owning root with a profile"
    if any(_kind(node) in {"Return_Stmt", "Stop_Stmt", "Error_Stop_Stmt", "Entry_Stmt"}
           for node in walk(owner.routine.execution)):
        return None, "root may leave before its single normal completion"
    plan = owner.execution_plan.validate()
    descriptors = []
    # Only descriptors already represented in this checked owner or required
    # original array formals are considered. No new source payload demand.
    roots = {binding.root for binding in owner.routine.scope.bindings.values()
             if binding.name in owner.routine.arguments and binding.rank}
    roots.update(owner.resources)
    for root in sorted(roots):
        try:
            binding = owner.builder.visible_binding(owner.routine, root)
            name = owner.builder.visible(owner.routine, root)
        except CompilationError:
            continue
        if (not binding.rank or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
                or binding.dtype not in {"integer", "real", "logical"}
                or binding.attributes & {"optional", "pointer", "volatile", "asynchronous", "value"}
                or len(binding.shape_nodes) != binding.rank
                or any(_kind(axis) not in {"Explicit_Shape_Spec", "Assumed_Shape_Spec", "Deferred_Shape_Spec"}
                       for axis in binding.shape_nodes)):
            continue
        # visible_binding resolves at the original procedure's entry scope,
        # never inside a later BLOCK/ASSOCIATE or through a section expression.
        active = owner.routine.scope
        while active is not None and active is not binding.declaring_scope:
            active = active.parent
        if active is None and (binding.declaring_scope is None
                               or _kind(binding.declaring_scope.node) not in {"Module", "Submodule"}):
            continue
        descriptors.append(Descriptor(binding, name))
    # The existing executable identity already binds source/configuration/facts
    # and the sealed emission inventory. Do not serialize transitive proofs a
    # second time solely to build a CPU performance key.
    static = (plan.execution_identity,
              _digest({"native_owner_hint_version": VERSION, "entry": owner.routine.qualified,
                       "profile": owner.builder.config.profile,
                       "entry_epoch": "reviewed facts only before first close; reopened NONE",
                       "definition_policy": "body effects with retained checked source admissions",
                       "descriptor_inventory": [item.binding.public() for item in descriptors]}))
    count = 4*len(static) + sum(2+2*item.binding.rank for item in descriptors)
    if count > MAX_WORDS:
        return None, "bounded entry metadata key capacity exceeded"
    identity = _digest({"static": static, "descriptor_roots": [item.binding.root for item in descriptors]})
    return NativeOwnerHint(owner, plan, identity, static, tuple(descriptors), count), None
