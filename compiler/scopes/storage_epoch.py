"""Storage lifetime contract derived from one sealed execution inventory.

This attachment authenticates canonical storage and the exact mandatory cuts.
It never grants definitions, source motion, or permission to cross a cut.
Physical generation/layout checks remain the runtime's responsibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256

from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def managed_resources(owner):
    """Canonical admitted inventory; no access or initialization authority."""
    managed = set()
    for member in owner.members.values():
        managed.update(member.canonical(binding).root for unit in member.units for binding in unit.arrays.values())
        from compiler.scopes.source_task_bindings import source_task_actions

        managed.update(root for task in member.source_device_tasks for root in source_task_actions(member, task))
        managed.update(member.canonical(mapping.binding).root for arrays in
                       getattr(member, "original_actual_registrations", {}).values() for mapping in arrays)
    return managed


def descriptor_windows(descriptor):
    """Original constraints carried by one complete graph lifetime contract."""
    from compiler.frontend.descriptor_windows import DescriptorWindowProof
    from compiler.frontend.supported_path_storage import SupportedPathDescriptorComposition
    from compiler.scopes.graph_regions import ControlLocalDescriptorRegions

    pending = [descriptor]
    found = {}
    while pending:
        proof = pending.pop()
        if type(proof) is DescriptorWindowProof:
            found[id(proof)] = proof
        elif type(proof) is SupportedPathDescriptorComposition:
            pending.extend(item[1] for group in
                           (proof._components, proof._exposures, proof._outside) for item in group)
            pending.extend(item[3] for item in proof._metadata)
        elif type(proof) is ControlLocalDescriptorRegions:
            pending.extend(item[1] for item in proof._components)
        else:
            raise CompilationError("private allocation obligations require their original descriptor consumer")
    return tuple(sorted(found.values(), key=lambda proof: (proof.procedure, proof.identity)))


def private_descriptor_windows(descriptor):
    return tuple(proof for proof in descriptor_windows(descriptor)
                 if proof._record.get("requires_unregistered_private_allocations"))



def worksharing_descriptor_windows(descriptor):
    """Only exact consumed windows, independent of private allocation roots."""
    return tuple(proof for proof in descriptor_windows(descriptor)
                 if proof._record.get("native_worksharing_storage_requirements"))


def _worksharing_windows(owner):
    epoch = getattr(owner, "position_epoch", None)
    return worksharing_descriptor_windows(epoch._descriptor) if epoch is not None else ()


def _retained_descriptor_windows(owner, proof_handoffs):
    """Inspect consumed proof payloads; never enumerate analysis registries."""
    from types import MappingProxyType

    from fparser.two.utils import Base

    from compiler.frontend.descriptor_windows import DescriptorWindowProof
    from compiler.frontend.source_effects import Binding, Routine, Scope, SourceEffects
    from compiler.scopes.source import ScopeBuilder

    pending = [getattr(owner, "return_contract", None), *(item.proof for item in proof_handoffs)]
    seen, found = set(), {}
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value) is DescriptorWindowProof:
            found[id(value)] = value
        elif isinstance(value, (Base, Binding, Routine, Scope, SourceEffects, ScopeBuilder)):
            continue
        elif isinstance(value, (dict, MappingProxyType)):
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, part.name) for part in fields(value))
    return tuple(sorted(found.values(), key=lambda proof: (proof.procedure, proof.identity)))


def _worksharing_rows(windows):
    from compiler.frontend.descriptor_worksharing import OriginalWorksharingStorage

    result = []
    for window in windows:
        tokens = window._worksharing
        if (not tokens or any(type(proof) is not OriginalWorksharingStorage
                or proof._analysis is not window._analysis
                or window._analysis.routines.get(proof._routine.qualified) is not proof._routine for proof in tokens)
                or [proof.public() for proof in tokens] !=
                    window.public()["native_worksharing_storage_requirements"]):
            raise CompilationError("worksharing descriptor lost its exact retained source requirements")
        public = window.public()
        result.append({"descriptor_identity": window.identity, "descriptor_procedure": window.procedure,
            "descriptor_structured_identity": public["structured_identity"],
            "descriptor_source_call_closure": public["source_call_closure"],
            "worksharing_requirements": [proof.public() for proof in tokens],
            "native_source_mode": "unchanged original callee; no generated body or companion",
            "internal_cuts_authorized": False, "execution_authorized": False,
            "payload_access_authorized": False, "entry_definition_authority": False})
    return result


def require_original_worksharing_coverage(parent, child):
    """A native projection cannot drop its parent's original source constraint."""
    if child is None or not child._record.get("native_worksharing_storage_requirements"):
        return child
    from compiler.scopes.graph_epochs import _same_source

    windows = worksharing_descriptor_windows(parent)
    if not windows or not all(_same_source(child._analysis, proof._analysis) for proof in windows):
        raise CompilationError("worksharing child lacks its authenticated original parent constraint")

    def source_key(record):
        return record["procedure"], _encoded(record["callee_source"]), _encoded(record["regions"])

    covered = {source_key(item) for row in _worksharing_rows(windows) for item in row["worksharing_requirements"]}
    required = {source_key(item) for item in child.public()["native_worksharing_storage_requirements"]}
    if not required <= covered:
        raise CompilationError("worksharing child changed or exceeds its parent original callee constraint")
    return child


def _overlap(first, last, span):
    # Empty insertions still alter the original line at the insertion boundary.
    return first <= span[1] and max(first, last) >= span[0]


def require_original_worksharing_edits(owner, final_builder_edits):
    """Reject source edits or managed members inside the exact native callee."""
    windows = _worksharing_windows(owner)
    requirements = [item for row in _worksharing_rows(windows) for item in row["worksharing_requirements"]]
    _require_original_worksharing_edits(owner, final_builder_edits, requirements)


def _require_original_worksharing_edits(owner, final_builder_edits, requirements):
    """Use detached source rows at the final callback-free boundary."""
    for requirement in requirements:
        source = requirement["callee_source"]
        for member in owner.members.values():
            if member.routine.qualified == requirement["procedure"]:
                raise CompilationError("worksharing whole native callee cannot be a managed member or companion")
            if str(member.routine.scope.path) != source["path"]:
                continue
            for unit in member.units:
                if _overlap(unit.scope.first, unit.scope.last, source["source_span"]):
                    raise CompilationError("worksharing managed unit overlaps a whole native callee")
            for task in (*member.device_tasks, *member.source_device_tasks, *member.storage_transfers):
                if _overlap(task.first, task.last, source["source_span"]):
                    raise CompilationError("worksharing managed task overlaps a whole native callee")
        for path, edits in final_builder_edits.items():
            if str(path) != source["path"]:
                continue
            for first, last, _replacement in edits:
                if _overlap(first, last, source["source_span"]):
                    raise CompilationError("worksharing source edit overlaps a whole unchanged native callee")


def _worksharing_inventory(owner, proof_handoffs=None):
    plan = getattr(owner, "execution_plan", None)
    handoffs = (getattr(plan, "_proof_handoffs", ()) if proof_handoffs is None else proof_handoffs)
    parents = _worksharing_windows(owner)
    consumed = tuple(proof for proof in _retained_descriptor_windows(owner, handoffs)
                     if proof._record.get("native_worksharing_storage_requirements"))
    for proof in consumed:
        if not parents:
            raise CompilationError("worksharing retained proof lacks its complete parent storage epoch")
        require_original_worksharing_coverage(owner.position_epoch._descriptor, proof)
    windows = tuple({id(proof): proof for proof in (*parents, *consumed)}.values())
    if not windows:
        return (), lambda: None, lambda: None
    attached = tuple((proof, tuple((part.name, getattr(proof, part.name)) for part in fields(proof)),
                      proof._analysis._descriptor_window_proofs) for proof in windows)
    checkpoints = tuple((token, token.checkpoint()) for proof in windows for token in proof._worksharing)
    requirements = _worksharing_rows(parents)
    source_requirements = tuple(item for row in requirements for item in row["worksharing_requirements"])
    _require_original_worksharing_edits(owner, owner.builder.edits, source_requirements)

    def checkpoint():
        current_parents = _worksharing_windows(owner)
        current_consumed = tuple(proof for proof in _retained_descriptor_windows(owner, handoffs)
                                 if proof._record.get("native_worksharing_storage_requirements"))
        current = tuple({id(proof): proof for proof in (*current_parents, *current_consumed)}.values())
        if (getattr(owner, "execution_plan", None) is not plan
                or (proof_handoffs is None and getattr(plan, "_proof_handoffs", ()) is not handoffs)
                or len(current) != len(windows) or any(a is not b for a, b in zip(current, windows, strict=True))
                or any(any(getattr(proof, name) is not value for name, value in saved)
                    or proof._analysis._descriptor_window_proofs is not registry
                    or registry.get(proof.identity) is not proof for proof, saved, registry in attached)):
            raise CompilationError("worksharing final source constraint attachment changed")
        for proof, saved in checkpoints:
            proof.require_checkpoint(saved)
        _require_original_worksharing_edits(owner, owner.builder.edits, source_requirements)

    def finish():
        for proof in windows:
            proof.validate(proof._analysis, proof.procedure, proof._selected)
        checkpoint()

    return requirements, finish, checkpoint


def _public_worksharing_checkpoint(builder):
    """Retain the published JSON objects without serializers or callbacks."""
    from compiler.scopes.storage_admission import _Identity

    scopes = builder.scopes
    pending = [scopes]
    seen, rows = set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value) is dict:
            items = tuple(dict.items(value))
            rows.append((_Identity(value), "dict", tuple((key, _Identity(item)) for key, item in items)))
            pending.extend(item for _key, item in items)
        elif type(value) in (list, tuple):
            items = tuple(value)
            rows.append((_Identity(value), "sequence", tuple(_Identity(item) for item in items)))
            pending.extend(items)
    return _Identity(scopes), tuple(rows)


def _require_public_worksharing_checkpoint(builder, saved):
    if _public_worksharing_checkpoint(builder) != saved:
        raise CompilationError("worksharing published scope or obligation changed during final validation")


def require_original_worksharing_publication(builder):
    """Final typed constraints after every scope has emitted its source edits."""
    retained = getattr(builder, "_original_worksharing_publications", ())
    if type(retained) is not tuple:
        raise CompilationError("worksharing final publication requires retained immutable epoch receipts")
    published = {scope["checked_execution_plan"]["storage_epoch"]["identity"]
        for scope in builder.scopes if isinstance(scope, dict) and "checked_execution_plan" in scope
        and scope["checked_execution_plan"].get("storage_epoch", {}).get("original_worksharing_storage_obligations")}
    if (any(type(proof) is not StorageEpochProof or proof._issuer is not _ISSUER for proof in retained)
            or len({proof.identity for proof in retained}) != len(retained)
            or {proof.identity for proof in retained} != published):
        raise CompilationError("worksharing published epoch lost or changed its retained source obligation")
    if not retained:
        return
    public_checkpoint = _public_worksharing_checkpoint(builder)
    receipts = tuple((proof, tuple((part.name, getattr(proof, part.name)) for part in fields(proof)),
                      proof._owner, proof._owner.execution_plan) for proof in retained)
    edits = builder.edits
    edit_rows = tuple((path, values, tuple(values)) for path, values in edits.items())
    final_checkpoints = []
    for proof, _fields, owner, plan in receipts:
        plan._require_owner()
        if (plan._storage_epoch is not proof or proof._owner is not owner
                or sha256(proof._record.encode()).hexdigest() != proof.identity):
            raise CompilationError("worksharing final epoch lost its checked owner/plan identity")
        # The owner-local capture context is correctly closed/restored now.
        # Revalidate source constraints, never recapture selected-window facts.
        current, finish, checkpoint = _worksharing_inventory(owner)
        final_checkpoints.append(checkpoint)
        if current != json.loads(proof._record).get("original_worksharing_storage_obligations"):
            raise CompilationError("worksharing final epoch source obligations changed")
        finish()
        _require_original_worksharing_edits(owner, edits,
            tuple(item for row in current for item in row["worksharing_requirements"]))
    builder.analysis.verify_source()
    builder.analysis.inputs.verify_live()
    for checkpoint in final_checkpoints:
        checkpoint()
    _require_public_worksharing_checkpoint(builder, public_checkpoint)
    if (getattr(builder, "_original_worksharing_publications", ()) is not retained
            or builder.edits is not edits or len(edits) != len(edit_rows)
            or any(edits.get(path) is not values or len(values) != len(items)
                or any(a is not b for a, b in zip(values, items, strict=True))
                for path, values, items in edit_rows)
            or any(any(getattr(proof, name) is not value for name, value in saved)
                or proof._owner is not owner or owner.execution_plan is not plan
                for proof, saved, owner, plan in receipts)):
        raise CompilationError("worksharing final publication or source edits changed")


def _private_descriptor_windows(owner):
    epoch = getattr(owner, "position_epoch", None)
    return private_descriptor_windows(epoch._descriptor) if epoch is not None else ()


def _private_allocation_inventory(owner):
    """Bind an obligation to the final complete canonical registration set.

    A private allocation may change while unrelated caller fields remain live.
    It must itself have no handle in that domain, including host-only alias
    reservations. Future checked children cannot register it by adding a fact.
    """
    windows = _private_descriptor_windows(owner)
    if not windows:
        return (), lambda: None
    from compiler.frontend.local_allocatable_storage import (
        allocation_source_checkpoint,
        require_allocation_source_checkpoint,
    )

    analyses = {id(proof._analysis): proof._analysis for proof in windows}
    source_receipts = tuple((analysis, allocation_source_checkpoint(analysis))
                            for analysis in analyses.values())
    resources = owner.resources
    epoch = owner.position_epoch
    descriptor = epoch._descriptor
    parents = tuple((value, tuple((part.name, getattr(value, part.name)) for part in fields(value)))
                    for value in (epoch, descriptor) if is_dataclass(value))
    entries = tuple((root, binding, number) for root, (binding, number) in resources.items())
    retained = tuple((proof, tuple((part.name, getattr(proof, part.name)) for part in fields(proof)),
                      proof._analysis._descriptor_window_proofs) for proof in windows)
    requirements = []
    for proof in windows:
        public = proof.public()
        roots = public["requires_unregistered_private_allocations"]
        if (not isinstance(roots, list) or not roots or roots != sorted(set(roots))
                or any(type(root) is not str for root in roots)):
            raise CompilationError("private allocation obligation requires exact canonical original roots")
        declarations = {item["resource"]: item["procedure"]
                        for item in public["native_private_allocation_requirements"]}
        if set(declarations) != set(roots):
            raise CompilationError("private allocation obligation lost an original declaring procedure")
        requirements.append({"descriptor_identity": proof.identity, "descriptor_procedure": proof.procedure,
            "unregistered_private_allocations": [
                {"resource": root, "declaring_procedure": declarations[root]} for root in roots]})
        if set(roots).intersection(resources):
            raise CompilationError("private allocation mutation overlaps a live canonical registration")

    def checkpoint():
        for analysis, saved in source_receipts:
            require_allocation_source_checkpoint(analysis, saved)
        current = _private_descriptor_windows(owner)
        if (len(current) != len(windows) or any(a is not b for a, b in zip(current, windows, strict=True))
                or owner.position_epoch is not epoch or epoch._descriptor is not descriptor
                or any(any(getattr(value, name) is not item for name, item in attached)
                       for value, attached in parents)
                or owner.resources is not resources or len(resources) != len(entries)
                or any(resources.get(root) is None or resources[root][0] is not binding
                       or resources[root][1] != number for root, binding, number in entries)
                or any(any(getattr(proof, name) is not value for name, value in attached)
                       or getattr(proof._analysis, "_descriptor_window_proofs", None) is not registry
                       or registry.get(proof.identity) is not proof for proof, attached, registry in retained)):
            raise CompilationError("private allocation obligation or final registration inventory changed")

    checkpoint()

    def finish():
        for proof in windows:
            proof.validate(proof._analysis, proof.procedure, proof._selected)
        for analysis in analyses.values():
            analysis.inputs.verify_live()
        checkpoint()

    return requirements, finish


def _record(owner, *, proof_handoffs=None):
    private_requirements, finish_private_inventory = _private_allocation_inventory(owner)
    worksharing_requirements, finish_worksharing_inventory, _checkpoint_worksharing = _worksharing_inventory(
        owner, proof_handoffs)
    managed, reservations = managed_resources(owner), set()
    for member in owner.members.values():
        reservations.update(member.canonical(binding).root for unit in member.units
                            for binding in (unit.host_metadata or {}).values())
        reservations.update(member.canonical(binding).root for unit in member.units for operation in unit.scope.native
                            for binding in operation.host_only_reads.values())
        reservations.update(member.canonical(mapping.binding).root for arrays in
                            getattr(member, "original_native_actual_registrations", {}).values() for mapping in arrays)
        reservations.update(member.canonical(element.binding).root for elements in
                            getattr(member, "original_scalar_element_reservations", {}).values() for element in elements)
    resources = []
    epoch = getattr(owner, "position_epoch", None)
    batch_bindings = tuple(binding for root, (binding, _number) in sorted(owner.resources.items()) if root in managed)
    capture_many = getattr(epoch, "capture_many", None)
    batched = None
    if (callable(capture_many) and getattr(owner.builder, "_graph_storage_epoch", None) is epoch
            and all(epoch.captures(binding) for binding in batch_bindings)):
        batched = dict(zip((binding.root for binding in batch_bindings),
                           capture_many(owner.builder, batch_bindings), strict=True))
    for root, (binding, number) in sorted(owner.resources.items()):
        record = {"resource": root, "registration_identity": number, "descriptor": binding.public()}
        if root in managed:
            fact = batched[root] if batched is not None else owner.builder.capture(binding)
            record.update(storage=fact["storage"], lifetime=fact.get("lifetime", "invocation"),
                          allocation_changes=fact["allocation_changes"], escapes=fact["escapes"],
                          device_replica_authorized=True)
        elif root in reservations:
            # Existing exact native proofs authorize an opaque alias range,
            # not a payload capture or a new GPU-readable allocation. Requiring
            # managed capture facts here would reject valid native-only inputs.
            record.update(storage="opaque host reservation", lifetime="original checked native access",
                          device_replica_authorized=False, initialization_authorized=False)
        else:
            raise CompilationError("storage epoch has an unclassified canonical registration: " + root)
        resources.append(record)
    members = []
    for member in owner.members.values():
        units = []
        for ordinal, unit in enumerate(member.units):
            original = member.builder.inline.original_selection(unit.nodes)
            graph, selected = member.builder.analysis._selected_source(member.routine.qualified, original)
            if graph is not member.graph or not selected:
                raise CompilationError("storage epoch requires the exact checked source selection")
            units.append({"ordinal": ordinal, "selected_node_ids": list(selected),
                          "canonical_resources": {root: member.canonical(binding).root
                                                   for root, binding in sorted(unit.arrays.items())}})
        record = {"procedure": member.routine.qualified,
                        "source_graph": member.graph.identity, "units": units,
                        "storage_transfers": [transfer.proof.validate(member.builder.analysis, member.routine,
                            transfer.node).public() for transfer in member.storage_transfers],
                        "retirement_cuts": [dict(item) for item in member.boundaries]}
        if member.device_tasks:
            record["device_tasks"] = [task.public(member) for task in member.device_tasks]
        if member.source_device_tasks:
            record["source_device_tasks"] = [task.public(member) for task in member.source_device_tasks]
        members.append(record)
    returned = getattr(owner, "return_contract", None)
    supported = bool(getattr(owner, "supported_path_mode", False))
    if supported and returned is not None:
        raise CompilationError("supported path and returned-domain lifetime contracts are distinct")
    record = {"version": 1, "entry": owner.routine.qualified,
            **({"return_contract": returned.validate().public()} if returned is not None else {}),
            **({("supported_path_epoch" if supported else "selected_position_epoch"): epoch.validate().public()}
               if epoch is not None else {}),
            "source_inputs": owner.builder.analysis.inputs.identity(),
            "resources": resources, "members": members,
            "activation": "first reached managed access; no descriptor evaluation on zero-trip paths",
            "lifetime": "checked original invocation between mandatory retirement cuts",
            "generation": "runtime-checked allocation generation; never inferred from address reuse",
            "readonly": "add a valid replica; preserve other current replicas",
            "writes": "invalidate only overlapping regions after ordered completion",
            "return": "checked children borrow; unmanaged return publishes observable storage",
            "definitions_authorized": False, "placement_authorized": False,
            "cross_cut_retention_authorized": False}
    if private_requirements:
        record.update(version=2, private_allocation_obligations=private_requirements)
        # All source/fact/epoch validators above may invoke callbacks. Publish
        # only after checking the original obligations and the same full map.
        finish_private_inventory()
    if worksharing_requirements:
        record.update(version=3, original_worksharing_storage_obligations=worksharing_requirements)
        finish_worksharing_inventory()
    return record


@dataclass(frozen=True, slots=True)
class StorageEpochProof:
    identity: str
    _owner: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, plan):
        plan._require_owner()
        if (self._issuer is not _ISSUER or self._owner is not plan._owner
                or plan._storage_epoch is not self):
            raise CompilationError("storage epoch requires its issuing checked execution plan")
        if _encoded(_record(self._owner)) != self._record:
            raise CompilationError("storage epoch source, storage facts or retirement cuts changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def admits(self, plan, member, binding):
        """Only exact checked unit bindings may borrow this epoch's storage."""
        return self.admits_many(plan, member, (binding,))[0]

    def admits_many(self, plan, member, bindings):
        """Validate the epoch once for one indivisible emission batch.

        This is not a validation cache: every batch rechecks the complete
        contract before inspecting its original binding objects.
        """
        self.validate(plan)
        if not any(current is member for current in plan.members):
            raise CompilationError("storage epoch does not own this original procedure")
        admitted = {id(value): value for unit in plan.units(member) for value in unit.arrays.values()}
        roots = []
        for binding in bindings:
            if admitted.get(id(binding)) is not binding:
                raise CompilationError("storage epoch requires an admitted original resource binding")
            root = member.canonical(binding).root
            if root not in self._owner.resources:
                raise CompilationError("storage epoch lost its canonical allocation binding")
            roots.append(root)
        return tuple(roots)


def issue_storage_epoch(owner, *, proof_handoffs=()):
    """Called by the execution-plan issuer only after its inventory is sealed."""
    encoded = _encoded(_record(owner, proof_handoffs=proof_handoffs))
    proof = StorageEpochProof(sha256(encoded.encode()).hexdigest(), owner, encoded, _ISSUER)
    if json.loads(encoded).get("original_worksharing_storage_obligations"):
        previous = getattr(owner.builder, "_original_worksharing_publications", ())
        if type(previous) is not tuple:
            raise CompilationError("worksharing publication requires immutable typed epoch receipts")
        owner.builder._original_worksharing_publications = (*previous, proof)
    return proof
