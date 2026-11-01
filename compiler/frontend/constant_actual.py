"""Recognize source-constant arithmetic without evaluating Fortran values.

Expressions remain at the original CALL and use the Fortran compiler's kinds
and constant folding. Mutable reads, inquiries and calls need separate original
position evaluation/coherence authority and are deliberately excluded here.
"""
from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.source_functions import _validate_scope
from compiler.scopes.immutable_inputs import _declared


def constant_scalar_actual(analysis, scope, expression):
    analysis.verify_source()
    _validate_scope(analysis, scope)
    visits = 0

    def constant(node, depth=0):
        nonlocal visits
        visits += 1
        if visits > analysis.operation_limit or depth > analysis.depth_limit:
            return False
        kind = _kind(node)
        if kind in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant",
                    "Signed_Int_Literal_Constant", "Signed_Real_Literal_Constant"}:
            return True
        if kind == "Name":
            original_scope = analysis.source_scope_for(node, scope)
            binding = analysis._binding(original_scope, node)
            if (binding is None or binding.rank or "parameter" not in binding.attributes
                    or binding.dtype not in {"integer", "real"}):
                return False
            _declared(analysis, binding)
            return True
        if kind not in {"Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand", "Mult_Operand", "Parenthesis"}:
            return False
        children = tuple(_children(node))
        return bool(children) and all(item in {"+", "-", "*", "/", "**", "(", ")"} if isinstance(item, str)
                                      else constant(item, depth + 1) for item in children)

    return constant(expression)
