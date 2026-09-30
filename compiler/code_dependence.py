from compiler.context import Context, Variable
from compiler.fparser_tree_abstraction import FparserTree


class Dependencies:
    def register_write_dependency(self, variable: Variable, vector: list[int]):
        pass

    def register_read_dependency(self, variable: Variable, vector: list[int]):
        pass


class CodeDependenceExtractor:
    def get_dependencies(self, code_ast, context: Context) -> Dependencies:

        deps = Dependencies()
        self._dispatch_node(code_ast, context, deps)

        return deps

    def _dispatch_node(self, node, context: Context, deps: Dependencies):
        node_type = node.__class__.__name__.lower()

        handle_functions = {
            "assignment_stmt": self._handle_assignment_stmt,
        }

        if node_type in handle_functions:
            handle_functions[node_type](node, context, deps)
        else:
            for child in FparserTree(node).children():
                self._dispatch_node(child, context, deps)

    def _handle_assignment_stmt(self, node, context: Context, deps: Dependencies):
        pass
