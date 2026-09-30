from __future__ import annotations
from decimal import Context


class FparserTree:
    def __init__(self, tree):
        if tree.__class__ == FparserTree:
            tree = tree.tree

        self.tree = tree

    def get_first_parent_of_type(self, node_type: str) -> FparserTree | None:
        current_node = self.tree

        while current_node is not None:
            if FparserTree(current_node).is_type(node_type):
                return FparserTree(current_node)
            
            if not hasattr(current_node, "parent"):
                return None
            
            current_node = current_node.parent

        return None

    def get_all_nodes_of_type(self, node_type: str) -> list:
        def add_if_node_is_of_type(node, acc):
            if node_type.lower() == node.__class__.__name__.lower():
                acc.append(node)
            return acc
        
        nodes = self.reduce_nodes(add_if_node_is_of_type, [])
        
        return nodes
    
    def get_all_nodes_in_children_of_type(self, node_type: str) -> list:
        if not hasattr(self.tree, "children"):
            return []
        
        return [child for child in self.tree.children if child.__class__.__name__.lower() == node_type.lower()]
    
    def get_first_child_of_type(self, node_type: str):
        all_nodes = self.get_all_nodes_in_children_of_type(node_type)

        if len(all_nodes) == 0:
            return None

        return all_nodes[0]

    def get_nodes(self, predicate) -> list:
        def add_if_node_satisfies_predicate(node, acc):
            if predicate(node):
                acc.append(node)
            return acc
        
        nodes = self.reduce_nodes(add_if_node_satisfies_predicate, [])
        
        return nodes

    def reduce_nodes(self, reduction_function, accumulator_init_value) -> list:
        return self._reduce_impl(self.tree, reduction_function, accumulator_init_value)

    def _reduce_impl(self, node, reduction_function, acc):
        acc = reduction_function(node, acc)

        if hasattr(node, "children"):
            for child in node.children:
                acc = self._reduce_impl(child, reduction_function, acc)

        return acc

    def children(self) -> list:
        if hasattr(self.tree, "children"):
            return [child for child in self.tree.children]
        else:
            return []
    
    def children_as_fTrees(self) -> list:
        if hasattr(self.tree, "children"):
            return [FparserTree(child) for child in self.tree.children]
        else:
            return []

    def is_type(self, node_type: str) -> bool:
        return self.tree.__class__.__name__.lower() == node_type.lower()

    def is_loop_definition(self) -> bool:
        return self.is_type("Block_Nonlabel_Do_Construct")

    def is_comment(self) -> bool:
        return self.is_type("Comment")

    def is_call_statement(self) -> bool:
        return self.is_type("Call_Stmt")

    def get_node_in_chain_of_types(self, node_idxs: list[int | str]):
        if len(node_idxs) == 0:
            return self.tree
        
        current_idx = node_idxs[0]

        if isinstance(current_idx, int):
            if hasattr(self.tree, "children"):
                child = self.tree.children[current_idx]
            else:
                child = self.tree[current_idx]
            return FparserTree(child).get_node_in_chain_of_types(node_idxs[1:])

        all_children = self.get_all_nodes_in_children_of_type(current_idx)

        if len(all_children) != 1:
            raise Exception(f"Expected exactly one child of type {current_idx}, but found {len(all_children)}")

        return FparserTree(all_children[0]).get_node_in_chain_of_types(node_idxs[1:])

    def get_children_without(self, types_to_exclude: list[str]) -> list:
        to_exclude = [t.lower() for t in types_to_exclude]
        return [
            child for child in self.children() 
            if child.__class__.__name__.lower() not in to_exclude]


class LoopStatement:
    def __init__(self, loop_ast):
        self.loop_ast = loop_ast

    def iteration_variable_name(self) -> str:
        path = ["Nonlabel_Do_Stmt", "Loop_Control", 1, 0]
        return str(FparserTree(self.loop_ast).get_node_in_chain_of_types(path))
    
    def get_execution_part(self):
        class GroupOfNodes:
            def __init__(self, nodes):
                self.children = nodes        

        if not hasattr(self.loop_ast, "children"):
            return GroupOfNodes([])

        do_loop_control_types = ["Nonlabel_Do_Stmt", "End_Do_Stmt"]
        return GroupOfNodes(FparserTree(self.loop_ast).get_children_without(do_loop_control_types))
    
    def range_code_ast_s(self) -> tuple[any, any, any]:
        loop_control_part = FparserTree(self.loop_ast).get_all_nodes_of_type("Loop_Control")[0]
        range_from_to_step = loop_control_part.children[1][1]
        range_from = range_from_to_step[0]
        range_to = range_from_to_step[1]
        step = range_from_to_step[2] if len(range_from_to_step) > 2 else None

        return range_from, range_to, step
    
    def get_loop_control_part(self):
        return FparserTree(self.loop_ast).get_all_nodes_of_type("Loop_Control")[0]


class CallStmtNode:
    def __init__(self, call_stmt_ast):
        self.call_stmt = FparserTree(call_stmt_ast)

    def called_function_name(self) -> str:
        return str(self.call_stmt.get_all_nodes_in_children_of_type("Name")[0])

    def get_arg_list(self, context: Context) -> list[Variable]:
        args_list = self.call_stmt.get_all_nodes_in_children_of_type("Actual_Arg_Spec_List")[0]
        args_list = FparserTree(args_list).children()
        args_list = [str(arg) for arg in args_list]

        return [context.get_variable_by_name(arg) for arg in args_list]