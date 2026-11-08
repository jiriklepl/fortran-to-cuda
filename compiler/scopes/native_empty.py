"""Retain a domain across an original CALL whose payload loops are empty.

This is a narrow source proof, not a native effect approximation. The original
CALL still executes once. Only side-effect-free descriptor inquiries move to
its entry; nonempty or unsupported work keeps the existing retirement route.
"""

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.native_guard import _association_proof, _formal_internal_child


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _body(nodes):
    result = []
    for node in nodes:
        if _kind(node) == "Comment":
            if str(node).lstrip().startswith("!$"):
                raise CompilationError("empty native CALL cannot move directive-controlled inquiries")
            continue
        if getattr(getattr(node, "item", None), "label", None) is not None:
            raise CompilationError("empty native CALL requires unlabeled structured control")
        result.append(node)
    return result


def _inquiry(analysis, routine, node, name):
    from compiler.frontend.source_functions import original_name_absent

    if (_kind(node) != "Intrinsic_Function_Reference" or str(node.items[0]).lower() != name
            or not original_name_absent(analysis, routine.scope, name)):
        raise CompilationError("empty native CALL requires an original intrinsic descriptor inquiry")
    actuals = _children(node.items[1])
    if len(actuals) != 1 or _kind(actuals[0]) != "Name":
        raise CompilationError("empty native CALL needs one whole named descriptor")
    return analysis._binding(routine.scope, actuals[0])


def _record(member, lifecycle):
    from compiler.frontend.resource_reachability import _host_arrange

    analysis = member.builder.analysis
    native = lifecycle._view._graph
    lifecycle._view.validate(analysis, member.routine.qualified, lifecycle._originals)
    if not lifecycle._view.public()["available"]:
        raise CompilationError("empty native CALL requires its complete same-family lifetime proof")
    roots = [i for i in native.host_instances if i.parent == 0]
    if len(roots) != 1 or roots[0].call.original is not lifecycle._originals[0]:
        raise CompilationError("empty native CALL lost its exact original invocation")
    root = roots[0]
    routine = analysis.routines[root.procedure]
    if routine.source_kind != "module":
        raise CompilationError("empty native CALL requires an original module wrapper")
    _association_proof(analysis, member.routine, routine, root.call.original, allow_contiguous_actual=True)
    # An INTENT(OUT) association invalidates values even if no body is reached.
    # Restrict the wrapper to descriptor-preserving ordinary numeric formals.
    forbidden = {"optional", "pointer", "allocatable", "value", "volatile", "asynchronous", "contiguous"}
    for name in routine.arguments:
        binding = routine.scope.bindings.get(name)
        if (binding is None or binding.intent == "out" or binding.attributes & forbidden
                or binding.dtype not in {"real", "integer", "logical"}
                or binding.rank and any(_kind(axis) != "Assumed_Shape_Spec" for axis in binding.shape_nodes)):
            raise CompilationError("empty native CALL has an unproved entry association")
    outer = dict(_host_arrange(root.call, routine))
    if any(_kind(actual) != "Name" for actual in outer.values()):
        raise CompilationError("empty native CALL requires whole original named actuals")
    calls = _body(_children(routine.execution))
    if not calls or len(calls) > analysis.operation_limit:
        raise CompilationError("empty native CALL has no bounded original child inventory")
    descriptors, routes = {}, []
    for call in calls:
        if _kind(call) != "Call_Stmt":
            raise CompilationError("empty native CALL wrapper contains observable non-call work")
        child = _formal_internal_child(analysis, routine, call)
        if child is None:
            raise CompilationError("empty native CALL requires its immediate original internal child")
        association = _association_proof(analysis, routine, child, call)
        matches = [i for i in native.host_instances if i.parent == root.ordinal and i.call.original is call]
        if len(matches) != 1 or matches[0].procedure != child.qualified:
            raise CompilationError("empty native CALL lost its exact child graph instance")
        executable = _body(_children(child.execution))
        if len(executable) != 1 or _kind(executable[0]) != "If_Construct":
            raise CompilationError("empty native CALL child needs one allocation guard")
        branch = _body(_children(executable[0]))
        if len(branch) != 3 or [_kind(n) for n in branch] != ["If_Then_Stmt", "Block_Nonlabel_Do_Construct", "End_If_Stmt"]:
            raise CompilationError("empty native CALL guard needs one counted payload loop")
        metadata = _inquiry(analysis, child, branch[0].items[0], "allocated")
        if (metadata is None or metadata.rank != 1 or "allocatable" not in metadata.attributes
                or metadata.attributes & {"pointer", "target", "optional", "volatile", "asynchronous", "value"}):
            raise CompilationError("empty native CALL needs an ordinary rank-one allocation descriptor")
        loop = _body(_children(branch[1]))
        if len(loop) < 2 or _kind(loop[0]) != "Nonlabel_Do_Stmt" or _kind(loop[-1]) != "End_Do_Stmt":
            raise CompilationError("empty native CALL requires original counted DO control")
        control = loop[0].items[1]
        if (_kind(control) != "Loop_Control" or control.items[0] is not None
                or control.items[1] is None or control.items[3] is not None):
            raise CompilationError("empty native CALL requires an ordinary counted iterator")
        index, bounds = control.items[1]
        iterator = analysis._binding(child.scope, index)
        if (iterator is None or iterator.rank or iterator.dtype != "integer" or iterator.attributes
                or iterator.declaring_scope is not child.scope or str(index).lower() in child.arguments):
            raise CompilationError("empty native CALL iterator must be an automatic private local")
        if (len(bounds) != 2 or _kind(bounds[0]) != "Int_Literal_Constant" or str(bounds[0]) != "1"
                or _inquiry(analysis, child, bounds[1], "size") is not metadata):
            raise CompilationError("empty native CALL needs the exact one-to-SIZE loop")
        actual = metadata
        if metadata.name in child.arguments:
            if metadata.intent != "in":
                raise CompilationError("empty native CALL descriptor formal must be read-only")
            args = dict(_host_arrange(matches[0].call, child))
            actual = analysis._binding(routine.scope, args[metadata.name])
        if actual is not None and actual.declaring_scope is routine.scope and actual.name in routine.arguments:
            actual = analysis._binding(member.routine.scope, outer[actual.name])
        if (actual is None or "allocatable" not in actual.attributes or actual.rank != 1
                or actual.attributes & {"pointer", "target", "optional", "volatile", "asynchronous", "value"}):
            raise CompilationError("empty native CALL descriptor lost its original association")
        imported = None
        try:
            name = member.builder.visible(member.routine, actual.root)
            visible = member.builder.visible_binding(member.routine, actual.root)
        except CompilationError:
            declaration = actual.declaring_scope
            if (analysis.modules.get(declaration.module) is not declaration
                    or declaration.bindings.get(actual.name) is not actual
                    or declaration.module == member.routine.scope.module
                    or not analysis._exported(declaration, actual.name)):
                raise CompilationError("empty native CALL cannot import a hidden or private descriptor") from None
            name = "fort_empty_meta_" + sha256(actual.root.encode()).hexdigest()[:12]
            imported = {"module": declaration.module, "remote": actual.name}
            visible = actual
        if (visible is not actual or not name.isidentifier()
                or name.lower() in {"allocated", "size"}
                or imported is None and name.lower().startswith("fort_")):
            raise CompilationError("empty native CALL descriptor is not directly visible at its original call")
        descriptors[actual.root] = {"resource": actual.root, "name": name, "import": imported}
        graph = analysis.structure(child.qualified)
        routes.append({"procedure": child.qualified, "source_graph": graph.identity,
                       "source_call": str(call), "association": association,
                       "original_guard": str(branch[0]), "original_loop": str(loop[0]),
                       "descriptor": actual.root})
    return {"version": 1, "kind": "empty_native_call_descriptor_guard",
            "procedure": member.routine.qualified, "call_preservation": lifecycle.identity,
            "native_graph_identity": native.identity, "descriptors": sorted(descriptors.values(), key=lambda r:r["resource"]),
            "routes": routes, "original_call_executes_once": True,
            "empty_path": "original descriptor guards and zero-trip loops; no payload or externally visible write",
            "nonempty_path": "existing full-domain retirement before unchanged original CALL",
            "payload_access_authorized": False, "entry_definition_authority": False,
            "allocation_state_assumed": False}


@dataclass(frozen=True)
class EmptyNativeCall:
    identity: str
    _member: object = field(repr=False, compare=False)
    _lifecycle: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)

    def validate(self, member, lifecycle):
        if (member is not self._member or lifecycle is not self._lifecycle
                or lifecycle._draft._empty_native_calls.get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity
                or _encode(_record(member, lifecycle)) != self._record):
            raise CompilationError("empty native CALL lost its original issued descriptor proof")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def wrap(self, close):
        flag = "fort_native_nonempty_" + self.identity[:12]
        descriptors = self.public()["descriptors"]
        imports = [f"use {d['import']['module']}, only: {d['name']} => {d['import']['remote']}"
                   for d in descriptors if d["import"] is not None]
        lines = ["block", *imports, "intrinsic :: allocated, size", "logical :: " + flag, flag + " = .false."]
        for descriptor in descriptors:
            name = descriptor["name"]
            lines += [f"if (allocated({name})) then", f"if (size({name}) /= 0) {flag} = .true.", "endif"]
        return [*lines, f"if ({flag}) then", *close, "endif", "end block"]


def prove_empty_native_call(member, lifecycle):
    from compiler.frontend.storage_mutation_effects import source_registry_budget

    record = _encode(_record(member, lifecycle))
    identity = sha256(record.encode()).hexdigest()
    registry = lifecycle._draft._empty_native_calls
    if identity not in registry:
        if len(registry) >= source_registry_budget(member.builder.analysis):
            raise CompilationError("empty native CALL source proof budget exhausted")
        registry[identity] = EmptyNativeCall(identity, member, lifecycle, record)
    return registry[identity].validate(member, lifecycle)
