"""Keep descriptor-only guards in an original, otherwise native child.

This is a continuation proof, not a complete effect summary or a GPU worker.
Whole descriptor associations may enter the original internal procedure. A
reached payload operation closes the outer owner before executing once.
"""

from copy import copy

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError, SourceLocation
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.segments import condition_fragment, fortran_lines, statement_span


def descriptor_loop_body(builder, node):
    """Keep a native counted loop's original header, including zero trips.

    Only scalar/descriptor control may run before publication. The graph
    authenticates that header separately from its unproved body. No bounds
    are copied, evaluated early, or evaluated a second time.
    """
    body = [item for item in node.content if _kind(item) != 'Comment']
    if (not body or _kind(body[0]) != 'Nonlabel_Do_Stmt'
            or _kind(body[-1]) != 'End_Do_Stmt'):
        raise CompilationError('native guarded loop requires original counted DO control')
    analysis, procedure = builder.analysis, builder.entry.qualified
    identity = analysis.structure(procedure).node_id(body[0], role='header')
    proof = analysis.segment_summary(procedure, (identity,), capture_locals=True)
    if (not proof['complete'] or proof.get('definition_changes')
            or proof.get('definition_diagnostics')
            or any(operation['kind'] not in {'read', 'write', 'overwrite', 'descriptor_read'}
                   or operation['rank'] and operation['kind'] != 'descriptor_read'
                   for operation in proof['operations'])):
        raise CompilationError('native guarded loop header needs scalar/descriptor-only effects')
    first, last = node.content.index(body[0]), node.content.index(body[-1])
    return node.content[first+1:last]


def _association_proof(analysis, caller, child, node):
    spec = _part(child.scope.node, 'Specification_Part')
    procedures = {}
    for declaration in _children(spec):
        if _kind(declaration) == 'Procedure_Declaration_Stmt':
            interface, attributes, entities = declaration.items
            if attributes is not None or _kind(interface) != 'Name':
                raise CompilationError('native guarded child needs ordinary source procedure formals')
            for entity in _children(entities):
                if _kind(entity) != 'Name':
                    raise CompilationError('native guarded procedure association is uncertain')
                procedures[str(entity).lower()] = str(interface).lower()
    values, position, keyword_seen = {}, 0, False
    for argument in _children(node.items[1]):
        if _kind(argument) == 'Actual_Arg_Spec':
            keyword_seen = True
            name, actual = str(argument.items[0]).lower(), argument.items[1]
        else:
            if keyword_seen or position >= len(child.arguments):
                raise CompilationError('native guarded call has unproved actual ordering')
            name, actual = child.arguments[position], argument
            position += 1
        if name not in child.arguments or name in values or _kind(actual) != 'Name':
            raise CompilationError('native guarded call needs whole named actuals')
        values[name] = actual
    if set(values) != set(child.arguments):
        raise CompilationError('native guarded child requires all original arguments')
    proof = []
    forbidden = {'optional', 'pointer', 'value', 'volatile', 'asynchronous', 'contiguous'}
    for name in child.arguments:
        actual = values[name]
        formal = child.scope.bindings.get(name)
        if name in procedures:
            targets = analysis._candidates(caller.scope, actual)
            if len(targets) != 1 or targets[0] not in analysis.numerical_helpers:
                raise CompilationError('native guarded procedure actual requires one source target')
            proof.append({'formal': name, 'actual': str(actual), 'source_target': targets[0],
                          'association': 'original procedure argument; never invoked before close'})
            continue
        binding = analysis._binding(caller.scope, actual)
        if (formal is None or binding is None or formal.signature() != binding.signature()
                or (formal.attributes | binding.attributes) & forbidden or formal.intent == 'out'
                or hasattr(binding, 'associate_selector')):
            raise CompilationError('native guarded actual association is unproved: ' + name)
        if formal.rank:
            if 'allocatable' in formal.attributes:
                if formal.intent != 'in' or 'allocatable' not in binding.attributes:
                    raise CompilationError('native guarded allocation descriptor must be read-only')
            elif (formal.dtype not in {'real', 'integer', 'logical'}
                  or any(_kind(axis) != 'Assumed_Shape_Spec' for axis in formal.shape_nodes)):
                raise CompilationError('native guarded array requires its original assumed-shape descriptor')
            else:
                for axis in formal.shape_nodes:
                    if axis.items[0] is not None:
                        child.scope.kinds.integer(axis.items[0], SourceLocation(str(child.scope.path)))
        elif formal.dtype not in {'real', 'integer', 'logical'} or 'allocatable' in formal.attributes:
            raise CompilationError('native guarded scalar association is unproved')
        proof.append({'formal': name, 'actual': str(actual), 'resource': binding.root,
                      'association': 'original descriptor or scalar reference; no synthetic dummy'})
    # Specification expressions execute before any body hook. Do not allow
    # an automatic bound, character length, or derived initialization here.
    for name, binding in child.scope.bindings.items():
        if name in child.arguments:
            continue
        if binding.dtype not in {'real', 'integer', 'logical'}:
            raise CompilationError('native guarded local initialization requires a separate proof')
        for axis in binding.shape_nodes:
            if _kind(axis) != 'Explicit_Shape_Spec':
                raise CompilationError('native guarded local descriptor lifetime is unproved')
            for bound in axis.items:
                if bound is not None:
                    child.scope.kinds.integer(bound, SourceLocation(str(child.scope.path)))
    return proof


class GuardedNativeChild(LexicalOwner):
    def scan(self, nodes, depth=0):
        if depth > self.builder.analysis.depth_limit:
            self.boundary(tuple(nodes), 'native guarded child control-depth budget exhausted')
            return
        for group in self.builder.inline.grouped_nodes(nodes):
            if isinstance(group, tuple):
                self.boundary(group, 'reached original native team; complete effects unavailable')
                continue
            kind = _kind(group)
            if kind in {'Comment', 'Continue_Stmt', 'Return_Stmt'}:
                if kind == 'Return_Stmt' and str(group).strip().lower() != 'return':
                    self.boundary((group,), 'unsupported alternate return')
                continue
            if kind == 'Block_Nonlabel_Do_Construct':
                try:
                    body = descriptor_loop_body(self.builder, group)
                except CompilationError as error:
                    self.boundary((group,), str(error))
                else:
                    self.scan(body, depth+1)
                continue
            if kind in {'If_Construct', 'If_Stmt'}:
                headers = ([item for item in group.content if _kind(item) in {'If_Then_Stmt', 'Else_If_Stmt'}]
                           if kind == 'If_Construct' else [group])
                try:
                    for header in headers:
                        if condition_fragment(self.builder, header).effects:
                            raise CompilationError('native child guard reads managed payload')
                except CompilationError as error:
                    self.boundary((group,), str(error))
                    continue
                if kind == 'If_Stmt':
                    self.boundary((group,), 'reached original native action; effects unavailable')
                else:
                    body = [item for item in group.content if _kind(item) not in {
                        'If_Then_Stmt', 'Else_If_Stmt', 'Else_Stmt', 'End_If_Stmt'}]
                    self.scan(body, depth+1)
                continue
            self.boundary((group,), 'reached original native operation; effects unavailable')


GUARDED_NATIVE_ROUTING_VERSION = 1


def _formal_internal_child(analysis, routine, node):
    """Recognize the exact original route used by the native-only scanner."""
    if _kind(node.items[0]) != 'Name':
        return None
    targets = analysis._candidates(routine.scope, node.items[0])
    if len(targets) != 1 or targets[0] not in analysis.routines:
        return None
    child = analysis.routines[targets[0]]
    if (child.source_kind != 'internal' or child.scope.parent is not routine.scope
            or not child.arguments):
        return None
    return child


def native_only_wrapper(parent, node, routine):
    """Fresh policy exclusion before any companion state is constructed.

    A formal-bearing immediate internal child only enters GuardedNativeChild,
    whose scanner never creates checked numerical/device work. The COMPLETE
    native graph still validates every callee, actual callback and normal-return
    effect; this routing choice grants no execution, capture or payload credit.
    Unknown source shapes retain ordinary checked-child preference.
    """
    from compiler.frontend.device_tasks import _task_contract
    from compiler.frontend.resource_reachability import prove_native_effect_graph
    from compiler.scopes.segments import directive

    builder, owner = parent.builder, parent.owner
    if builder.analysis.routines.get(routine.qualified) is not routine:
        raise CompilationError("native-only wrapper requires its exact original routine")
    if (GUARDED_NATIVE_ROUTING_VERSION != 1 or builder.config.scope_execution != "graph"
            or builder.config.policy != "sections" or builder.config.collective
            or owner.interval_draft is not None or parent.interval_draft is not None
            or getattr(owner.builder, "_conditional_return_draft", None) is None
            or getattr(builder, "_reached_window_binding", None) is not None
            or getattr(builder, "_mapped_reached_child", None) is not None
            or getattr(owner.builder, "provider_only_lifetime", None) is not None
            or routine.source_kind != "module" or routine.qualified in owner.members):
        return False
    originals = builder.inline.original_selection((node,))
    if len(originals) != 1 or originals[0] is not node or _kind(node) != "Call_Stmt":
        return False
    calls, visits = [], 0

    def sequence(nodes, depth=0):
        nonlocal visits
        if depth > builder.analysis.depth_limit:
            return False
        for item in nodes:
            visits += 1
            if visits > builder.analysis.operation_limit:
                return False
            kind = _kind(item)
            if kind == "Comment":
                if directive(item) is not None:
                    return False
            elif kind in {"Continue_Stmt", "Return_Stmt"}:
                if kind == "Return_Stmt" and any(part is not None for part in _children(item)):
                    return False
            elif kind == "Call_Stmt":
                if _task_contract(builder.analysis, routine, item) is not None:
                    return False  # Existing external-provider routing has priority.
                child = _formal_internal_child(builder.analysis, routine, item)
                if child is None:
                    return False
                calls.append((item, child))
            elif kind == "If_Construct":
                body = tuple(part for part in _children(item) if _kind(part) not in {
                    "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"})
                if not sequence(body, depth + 1):
                    return False
            elif kind == "If_Stmt":
                if not sequence((item.items[1],), depth + 1):
                    return False
            else:
                return False  # In particular local loops/array work/other checked children.
        return True

    if not sequence(_children(routine.execution)) or not calls:
        return False
    owner.builder._conditional_return_draft.validate()
    native = prove_native_effect_graph(builder.analysis, parent.routine.qualified, originals, project_payload=False)
    view = native.same_family_call_view()
    if not view.public()["available"]:
        return False
    roots = [instance for instance in native.host_instances if instance.parent == 0
             and instance.call.original is node and not instance.call.function]
    if len(roots) != 1 or roots[0].procedure != routine.qualified:
        raise CompilationError("native-only wrapper lost its exact original outer CALL")
    for original, child in calls:
        matches = [instance for instance in native.host_instances if instance.parent == roots[0].ordinal
                   and instance.call.original is original and not instance.call.function]
        if len(matches) != 1 or matches[0].procedure != child.qualified:
            raise CompilationError("native-only wrapper lost its exact original formal child route")
    view.validate(builder.analysis, parent.routine.qualified, originals)
    return True


def borrow(parent, node):
    """Retain original formal associations; no effect or alias guess is used."""
    analysis = parent.builder.analysis
    child = _formal_internal_child(analysis, parent.routine, node)
    if child is None:
        return False
    owner = parent.owner
    if child.qualified in owner.active:
        raise CompilationError('recursive native guarded child')
    proof = _association_proof(analysis, parent.routine, child, node)
    if child.qualified not in owner.members:
        if len(owner.members) >= analysis.procedure_limit:
            raise CompilationError('native guarded child procedure budget exhausted')
        from compiler.scopes.region_dispatch import InlineRegions
        proxy = copy(parent.builder)
        proxy.entry = child
        proxy.inline = InlineRegions(proxy)
        nested = GuardedNativeChild(proxy, parent=parent)
        nested.scan(_children(child.execution))
        owner.members[child.qualified] = nested
    elif not isinstance(owner.members[child.qualified], GuardedNativeChild):
        raise CompilationError('native guarded child conflicts with another source implementation')
    first, last = statement_span(node)
    record = {'procedure': child.qualified, 'caller': parent.routine.qualified,
              'first_line': first, 'last_line': last, 'association_proof': proof,
              'execution': 'original guarded native child; publish and close before reached payload work',
              'effect_summary_available': False}
    parent.internal_calls.append(record)
    parent.builder.resolved_calls[(parent.routine.qualified, (first, last))] = record
    from compiler.scopes.provenance import borrowed
    parent.patches.append((first, last, "\n".join(fortran_lines(borrowed(parent, (node,),
        parent.original((node,)).splitlines()))) + "\n"))
    return True
