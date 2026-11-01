"""Original OPTIONAL output storage, without reading an absent argument.

This is a descriptor-only policy. It does not make an optional payload a
capture, authorize its evaluation, or prove that an output is initialized.
"""


def optional_numeric_output(routine, binding):
    """Recognize only an exact ordinary scalar or explicit-shape vector formal."""
    return (binding is not None and binding.name in routine.arguments
            and routine.scope.bindings.get(binding.name) is binding
            and binding.declaring_scope is routine.scope and binding.intent == "out"
            and "optional" in binding.attributes and binding.rank in {0, 1}
            and binding.dtype in {"real", "integer"} and binding.kind in {4, 8}
            and not binding.attributes & {"target", "pointer", "allocatable", "volatile", "asynchronous", "value"}
            and (not binding.rank or len(binding.shape_nodes) == 1
                 and type(binding.shape_nodes[0]).__name__ == "Explicit_Shape_Spec"))


def independent_scalar_output(check, routine, binding):
    """Admit no aliased/dummy/module storage under the new writable policy."""
    if (binding is None or binding.rank or binding.declaring_scope is not routine.scope
            or binding.name in routine.arguments or binding.intent is not None
            or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
            or binding.attributes & {"target", "pointer", "allocatable", "optional", "parameter",
                                     "volatile", "asynchronous", "value"}):
        return False
    from compiler.frontend.source_functions import _walk

    specification = next((part for part in getattr(routine.scope.node, "content", ())
                          if type(part).__name__ == "Specification_Part"), None)
    for node in _walk(specification):
        check.tick()
        if type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt"}:
            return False
    return True


def _present(check, routine, node, binding):
    if type(node).__name__ != "Intrinsic_Function_Reference" or str(node.items[0]).lower() != "present":
        return False
    scope = check.analysis.source_scope_for(node, routine.scope)
    if (check.analysis._binding(scope, "present") or check.analysis._candidates(scope, "present")
            or check.analysis._unknown_exports(scope)):
        return False
    args = tuple(node.items[1].items)
    if len(args) != 1:
        return False
    argument = args[0]
    if type(argument).__name__ == "Actual_Arg_Spec":
        if str(argument.items[0]).lower() != "a":
            return False
        argument = argument.items[1]
    return type(argument).__name__ == "Name" and check.analysis._binding(scope, argument) is binding


def original_positive_presence(check, routine, node, binding):
    """Check exact lexical PRESENT use/then dominance, without evaluating it.

    Every parent edge must also retain the child as an original parser child;
    merely assigning a foreign parent pointer cannot supply a guard. The full
    chain must end at this authenticated routine's original execution part.
    ELSE and ELSE IF arms never borrow the first arm's positive condition.
    """
    current, guarded = node, False
    while current is not routine.execution:
        check.tick()
        parent = getattr(current, "parent", None)
        children = (getattr(parent, "content", getattr(parent, "items", ())) or ())
        if parent is None or not any(child is current for child in children):
            return False
        kind = type(parent).__name__
        if kind == "Intrinsic_Function_Reference" and _present(check, routine, parent, binding):
            guarded = True  # PRESENT itself does not evaluate its optional payload.
        elif kind == "If_Stmt" and parent.items[1] is current and _present(check, routine, parent.items[0], binding):
            guarded = True
        elif kind == "If_Construct":
            index = next(index for index, child in enumerate(children) if child is current)
            header = children[0]
            if (index > 0 and type(header).__name__ == "If_Then_Stmt"
                    and not any(type(child).__name__ in {"Else_Stmt", "Else_If_Stmt"} for child in children[1:index + 1])
                    and _present(check, routine, header.items[0], binding)):
                guarded = True
        current = parent
    return guarded
