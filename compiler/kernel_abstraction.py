from __future__ import annotations

from typing import Generator

from compiler.context import Context, ContextWithArguments, DoLoopContext, LocalContext, Variable
from compiler.fparser_tree_abstraction import CallStmtNode, FparserTree, LoopStatement
from compiler.debugging.color_printer import Colors as c


class Kernel:
    def __init__(self, context: Context):
        self._code_lines = []
        self.context = context
        self.sub_kernel = None

    def append_code_line(self, line_ast):
        self._code_lines.append(line_ast)

    def is_empty(self):
        return len(self._code_lines) == 0 and self.sub_kernel is None

    def has_top_level_context(self):
        return self.context.is_call_context()

    def merge_with(self, other_kernel):
        if self.sub_kernel is not None:
            raise Exception("Cannot merge with another kernel because this kernel already has a sub-kernel")

        self.sub_kernel = other_kernel

    def enum_lines_with_context(self):
        for line in self._code_lines:
            yield line, self.context

        if self.sub_kernel is not None:
            yield from self.sub_kernel.enum_lines_with_context()

    def enum_do_stmt_ranges_with_context(self) -> Generator[(any, DoLoopContext)]:
        for do_loop_context in self.context.enum_do_loop_contexts():
            for range_code in do_loop_context.enum_range_code():
                yield range_code, do_loop_context

    def get_loop_depth(self) -> int:
        self.context.enum_do_loop_contexts()
        return len(list(self.context.enum_do_loop_contexts()))

    def __str__(self):
        code_str = "\n".join([f"{c.CODE}{str(line)}{c.END}" for line in self._code_lines])
        tabbed_code_str = "\t" + code_str.replace("\n", "\n\t")

        context_str = str(self.context)
        tabbed_context_str = "\t" + context_str.replace("\n", "\n\t")

        sub_kernel_str = str(self.sub_kernel) if self.sub_kernel is not None else f"{c.NONE}None{c.END}"
        sub_kernel_tabbed = "\t" + sub_kernel_str.replace("\n", "\n\t")

        return f"{c.CLASS}Kernel{c.END}(\n  {c.FIELD}context{c.END}=\n{tabbed_context_str},\n  {c.FIELD}code_lines{c.END}=\n{tabbed_code_str}\n  {c.FIELD}sub_kernel{c.END}=\n{sub_kernel_tabbed}\n)"

    def get_all_do_loop_contexts_from_outer_to_inner(self) -> list[DoLoopContext]:
        contexts = list(self.context.enum_do_loop_contexts())
        contexts.reverse()
        return list(contexts)

    def get_all_do_loop_contexts_from_inner_to_outer(self) -> list[DoLoopContext]:
        return list(self.context.enum_do_loop_contexts())


class KernelFunctionDefinition:
    def __init__(self, kernel_ast):
        self.full_tree = FparserTree(kernel_ast)

        self.symbol_table: dict[str, "KernelFunctionDefinition"] = None

        self.declaration_ast = FparserTree(self.full_tree.get_all_nodes_in_children_of_type("Subroutine_Stmt")[0])
        self.specification_ast = FparserTree(self.full_tree.get_all_nodes_in_children_of_type("Specification_Part")[0])
        self.code_ast = FparserTree(self.full_tree.get_all_nodes_in_children_of_type("Execution_Part")[0])

        self.local_context = LocalContext(self.specification_ast, self.declaration_ast)

    def get_module_name(self) -> str | None:

        module_node = self.full_tree.get_first_parent_of_type("Module")

        if module_node is None:
            raise Exception("Expected the kernel function to be defined inside a module, but no parent module was found")

        module_stmt = FparserTree(module_node).get_all_nodes_of_type("Module_Stmt")[0]
        name = FparserTree(module_stmt).get_all_nodes_in_children_of_type("Name")[0]

        return str(name)

    def set_symbol_table(self, symbol_table: dict[str, "KernelFunctionDefinition"]):
        self.symbol_table = symbol_table

    def name(self):
        return str(self.declaration_ast.get_all_nodes_in_children_of_type("Name")[0])

    def extract_kernels_graph(self) -> list[Kernel]:
        return self._extract_kernels_sub_graph(self.code_ast.tree, self.local_context)

    def extract_kernels_graph_with_calls(self, caller_context: Context, call_args: list[Variable]) -> list[Kernel]:
        context_with_args = ContextWithArguments(self.local_context, caller_context, call_args)
        return self._extract_kernels_sub_graph(self.code_ast.tree, context_with_args)

    def parameters(self) -> list[Variable]:
        return [var for var in self.local_context.variables if var.is_function_param()]

    def _extract_kernels_sub_graph(self, code_ast, current_context) -> list[Kernel]:

        class CurrentKernels:
            def __init__(self):
                self.kernels = []
                self.current_sub_kernel = Kernel(current_context)

            def add_code_line(self, line_ast):
                self.current_sub_kernel.append_code_line(line_ast)

            def finish_current_sub_kernel(self):
                if not self.current_sub_kernel.is_empty():
                    self.kernels.append(self.current_sub_kernel)
                    self.current_sub_kernel = Kernel(current_context)

        current_kernels = CurrentKernels()
        lines_of_code = FparserTree(code_ast).children_as_fTrees()

        for line in lines_of_code:
            if line.is_comment():
                continue

            elif line.is_loop_definition():
                current_kernels.finish_current_sub_kernel()

                loop_statement = LoopStatement(line.tree)
                loop_context = DoLoopContext(loop_statement, current_context)

                execution_part = loop_statement.get_execution_part()

                sub_kernels = self._extract_kernels_sub_graph(execution_part, loop_context)
                current_kernels.kernels.extend(sub_kernels)

            elif line.is_call_statement():
                call = CallStmtNode(line)

                kernel_func_name = call.called_function_name()
                arg_list = call.get_arg_list(current_context)

                called_kernel = self.symbol_table[kernel_func_name]
                sub_kernels = called_kernel.extract_kernels_graph_with_calls(current_context, arg_list)

                if len(sub_kernels) > 0 and sub_kernels[0].has_top_level_context():
                    current_kernels.current_sub_kernel.merge_with(sub_kernels[0])
                    sub_kernels = sub_kernels[1:]

                if len(sub_kernels) > 0:
                    current_kernels.finish_current_sub_kernel()
                    current_kernels.kernels.extend(sub_kernels)

            else:
                current_kernels.add_code_line(line.tree)

        current_kernels.finish_current_sub_kernel()
        return current_kernels.kernels
