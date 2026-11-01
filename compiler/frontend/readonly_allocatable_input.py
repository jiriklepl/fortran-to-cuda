"""Original nonescaping ALLOCATABLE INTENT(IN) dummy association.

This is descriptor/association evidence, not a payload effect summary. Fortran
2018 8.5.10 and 15.5.2.13 require allocation changes to occur through the dummy
and forbid them for INTENT(IN). The complete original use projection below
separately excludes exporting this dummy's address or association. Unrelated
native effects still need their ordinary lifetime/coherence boundaries.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.native_group_structure import directive
from compiler.frontend.source_functions import (
    _scope_roles,
    _validate_resolution,
    _validate_scope,
    _validate_scope_roles,
)
from compiler.frontend.storage_component_reads import _read_position
from compiler.ir import CompilationError

_ISSUER = object()
_UNSAFE = {"pointer", "target", "optional", "volatile", "asynchronous", "value", "save", "parameter"}
_ALIASES = {"Namelist_Stmt", "Common_Stmt", "Equivalence_Stmt", "Cray_Pointer_Stmt",
            "Codimension_Stmt", "Codimension_Attr_Spec", "Bind_Stmt"}


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _walk(root):
    # The inventory is the complete authenticated original procedure, never a
    # numerical optimizer window or a caller-supplied work allowance.
    pending = [(root, ())]
    while pending:
        node, parents = pending.pop()
        if not isinstance(node, (Base, tuple, list)):
            continue
        yield node, parents
        pending.extend((child, (*parents, (node, index)))
                       for index, child in reversed(tuple(enumerate(_children(node)))))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sharing(analysis, routine, binding, syntax):
    """No task/address export or replacement of the original dummy storage."""
    texts, stack = [], []
    for node, _parents in syntax:
        text = directive(node)
        if text is None:
            if _kind(node) == "Comment" and str(node).lstrip().startswith("!$"):
                raise CompilationError("readonly allocatable input requires original joined concurrency")
            continue
        texts.append(text)
        opening = re.fullmatch(r"(parallel do|parallel|do)(?:\s+(.*))?", text)
        if opening:
            kind, suffix = opening.groups()
            if (kind.startswith("parallel") and stack) or (kind == "do" and stack != ["parallel"]):
                raise CompilationError("readonly allocatable input requires complete original joined teams")
            remainder = suffix or ""
            while remainder:
                clause = re.match(r"(private|shared|default|collapse|schedule|reduction)\s*\(([^()]*)\)\s*", remainder)
                if clause is None:
                    raise CompilationError("readonly allocatable input has unproved concurrency storage")
                name, values = clause.group(1), clause.group(2).strip()
                if name in {"private", "shared"}:
                    for spelling in values.split(","):
                        spelling = spelling.strip()
                        if not re.fullmatch(r"[a-z][a-z0-9_]*", spelling):
                            raise CompilationError("readonly allocatable input requires original sharing bindings")
                        shared = analysis._binding(routine.scope, spelling)
                        if shared is None or (shared is binding and name != "shared"):
                            raise CompilationError("readonly allocatable input cannot privatize its original descriptor")
                elif name == "reduction":
                    operator, separator, names = values.partition(":")
                    if separator != ":" or operator.strip() not in {
                            "+", "-", "*", "max", "min", ".and.", ".or.", ".eqv.", ".neqv.", "iand", "ior", "ieor"}:
                        raise CompilationError("readonly allocatable input has unproved reduction storage")
                    for spelling in names.split(","):
                        spelling = spelling.strip()
                        reduced = analysis._binding(routine.scope, spelling)
                        if (not re.fullmatch(r"[a-z][a-z0-9_]*", spelling) or reduced is None
                                or reduced is binding or reduced.declaring_scope is not routine.scope
                                or reduced.name in routine.arguments or reduced.rank
                                or reduced.dtype not in {"real", "integer", "logical"}
                                or reduced.attributes & (_UNSAFE | {"allocatable"})):
                            raise CompilationError("readonly allocatable input requires independent scalar reduction storage")
                elif name == "default" and values not in {"shared", "none"}:
                    raise CompilationError("readonly allocatable input cannot replace default storage association")
                elif name == "collapse" and values not in {"1", "2", "3", "4"}:
                    raise CompilationError("readonly allocatable input requires literal original collapse metadata")
                elif name == "schedule" and not re.fullmatch(
                        r"(?:static|runtime|dynamic|guided)(?:\s*,\s*[1-9][0-9]*)?", values):
                    raise CompilationError("readonly allocatable input has unproved original schedule metadata")
                remainder = remainder[clause.end():].lstrip(", ")
            stack.append(kind)
        elif text in {"end parallel", "end parallel do", "end do", "end do nowait"}:
            expected = "do" if text.startswith("end do") else text[4:]
            if not stack or stack.pop() != expected:
                raise CompilationError("readonly allocatable input changed its original team join")
        elif text == "barrier" and stack == ["parallel"]:
            pass
        else:
            raise CompilationError("readonly allocatable input has unproved asynchronous concurrency")
    # Require explicit closing directives in this first association slice.
    if stack:
        raise CompilationError("readonly allocatable input requires its complete parent join")
    return texts


def _derive(analysis, routine, binding):
    if (getattr(analysis, "_descriptor_source", None) is not None
            or analysis._require_original(routine.qualified) is not routine):
        raise CompilationError("readonly allocatable input requires its exact original procedure")
    inputs, input_identity = analysis.inputs, analysis.inputs.identity()
    roles, scopes = _scope_roles(analysis), {}

    def scope_authority(scope):
        _validate_scope(analysis, scope)
        scopes[id(scope)] = scope

    scope_authority(routine.scope)
    if (routine.scope.bindings.get(binding.name) is not binding or binding.declaring_scope is not routine.scope
            or binding.name not in routine.arguments or binding.intent != "in"
            or "allocatable" not in binding.attributes or binding.attributes & _UNSAFE
            or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {1, 2, 4, 8}
            or not 1 <= binding.rank <= 3):
        raise CompilationError("readonly allocatable input requires an exact independent numeric ALLOCATABLE INTENT(IN) dummy")
    syntax = tuple(_walk(routine.scope.node))
    if any(_kind(node) == "Internal_Subprogram_Part" for node, _parents in syntax):
        raise CompilationError("readonly allocatable input cannot escape through contained host association")
    specification = next((node for node, _parents in syntax if _kind(node) == "Specification_Part"), None)
    spec = tuple(_walk(specification))
    if any(_kind(node) in _ALIASES or (_kind(node) == "Comment" and str(node).lstrip().startswith("!$"))
           for node, _parents in spec):
        raise CompilationError("readonly allocatable input requires independent original declaration storage")
    declarations = [node for node, _parents in spec if _kind(node) == "Entity_Decl"
                    and str(node.items[0]).lower() == binding.name]
    if (len(declarations) != 1 or declarations[0].items[2] is not None or declarations[0].items[3] is not None
            or len(binding.shape_nodes) != binding.rank
            or any(_kind(axis) not in {"Deferred_Shape_Spec", "Assumed_Shape_Spec"}
                   or any(bound is not None for bound in axis.items) for axis in binding.shape_nodes)):
        raise CompilationError("readonly allocatable input requires its original deferred-shape declaration")
    for node, parents in spec:
        if (_kind(node) == "Name" and str(node).lower() == binding.name
                and (not parents or parents[-1][0] is not declarations[0] or parents[-1][1] != 0)):
            raise CompilationError("readonly allocatable input specification use requires separate association authority")
    execution = tuple(_walk(routine.execution))
    directives = _sharing(analysis, routine, binding, execution)
    uses = []
    for node, parents in execution:
        if _kind(node) != "Name" or str(node).lower() != binding.name:
            continue
        scope = analysis.source_scope_for(node, routine.scope)
        _validate_resolution(analysis, scope, (node,), _scope_validator=scope_authority)
        if analysis._binding(scope, node) is not binding:
            continue
        if any(_kind(parent) in {"Associate_Construct", "Select_Type_Construct", "Select_Rank_Construct",
                                 "Data_Ref"} for parent, _branch in parents):
            raise CompilationError("readonly allocatable input has an unproved alias or component observer")
        position = _read_position(analysis, routine, node, lambda: None)
        if len(position) != len(parents) or any(a is not b or i != j for (a, i), (b, j) in zip(position, parents, strict=True)):
            raise CompilationError("readonly allocatable input original occurrence changed")
        uses.append((node, parents))
    _validate_scope_roles(analysis, roles, scopes)
    for scope in tuple(scopes.values()):
        scope_authority(scope)
    if analysis._require_original(routine.qualified) is not routine:
        raise CompilationError("readonly allocatable input original source authority changed")
    record = {"version": 1, "kind": "original_readonly_allocatable_input", "procedure": routine.qualified,
              "resource": binding.root, "source_inputs": input_identity, "original_binding": binding.public(),
              "original_procedure_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
              "original_read_occurrences": len(uses), "original_directives": directives,
              "descriptor_rule": "Fortran 2018 8.5.10 and 15.5.2.13: allocation status changes only through INTENT(IN) dummy",
              "direct_use_projection": "complete original declaration and executable uses; no address or association export",
              "requires_independent_whole_allocatable_actual": True,
              "requires_reached_allocation_presence_and_alias_guards": True,
              "unrelated_native_effects_require_ordinary_boundaries": True,
              "allocation_presence_authority": False, "payload_definition_authority": False,
              "payload_access_authorized": False, "execution_authorized": False,
              "capture_authorized": False, "cross_frontier_retention_authorized": False}
    encoded = _encoded(record)
    # Serialization/comparison is not a publication fence. Reauthenticate the
    # consumed occurrences and live inputs after every callback has returned.
    for node, parents in uses:
        current = _read_position(analysis, routine, node, lambda: None)
        if len(current) != len(parents) or any(a is not b or i != j for (a, i), (b, j) in zip(current, parents, strict=True)):
            raise CompilationError("readonly allocatable input consumed occurrence changed before publication")
    _validate_scope_roles(analysis, roles, scopes)
    for scope in scopes.values():
        scope_authority(scope)
    if analysis.inputs is not inputs or analysis.inputs.identity() != input_identity:
        raise CompilationError("readonly allocatable input source inputs changed before publication")
    analysis.verify_publication()
    return encoded, tuple(uses)


def _same_uses(left, right):
    return len(left) == len(right) and all(a is b and len(ap) == len(bp)
        and all(x is y and i == j for (x, i), (y, j) in zip(ap, bp, strict=True))
        for (a, ap), (b, bp) in zip(left, right, strict=True))


@dataclass(frozen=True, slots=True)
class ReadonlyAllocatableInputProof:
    identity: str
    binding: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _uses: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, routine, binding):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or binding is not self.binding
                or getattr(analysis, "_readonly_allocatable_inputs", {}).get(self.identity) is not self):
            raise CompilationError("readonly allocatable input requires its exact compiler-issued proof")
        record, uses = _derive(analysis, routine, binding)
        if record != self._record or not _same_uses(uses, self._uses):
            raise CompilationError("readonly allocatable input source, binding or occurrence authority changed")
        if getattr(analysis, "_readonly_allocatable_inputs", {}).get(self.identity) is not self:
            raise CompilationError("readonly allocatable input proof registry changed before publication")
        analysis.verify_publication()
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def prove_readonly_allocatable_input(analysis, routine, binding):
    registry = getattr(analysis, "_readonly_allocatable_inputs", None)
    if registry is None:
        registry = analysis._readonly_allocatable_inputs = {}
    for proof in tuple(registry.values()):
        if proof._routine is routine and proof.binding is binding:
            return proof.validate(analysis, routine, binding)
    record, uses = _derive(analysis, routine, binding)
    if getattr(analysis, "_readonly_allocatable_inputs", None) is not registry:
        raise CompilationError("readonly allocatable input proof registry changed before publication")
    proof = ReadonlyAllocatableInputProof(sha256(record.encode()).hexdigest(), binding, analysis, routine,
                                        record, uses, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate(analysis, routine, binding)
