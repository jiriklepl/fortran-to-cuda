from compiler.fparser_tree_abstraction import FparserTree

from .kernel_abstraction import KernelFunctionDefinition
from fparser.two.parser import ParserFactory
from fparser.common.readfortran import FortranFileReader
import os


class SourceFilesCollection_FromFilesystem:
    def __init__(self):
        self.files_paths = []

    def load_dir(self, path, files_extensions):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(files_extensions):
                    self.files_paths.append(os.path.join(root, file))
        return self

    def load_file(self, path):
        self.files_paths.append(path)
        return self

    def iterate_paths(self, first_line_predicate=None):
        for file_path in self.files_paths:
            if first_line_predicate is None:
                yield file_path
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip().lower()

                if first_line_predicate(first_line):
                    yield file_path

class KernelFinder:
    def __init__(self, source_files_collection):
        self.source_files = source_files_collection
        self.kernels_dict = {}

    def load_all_kernels(self):
        def is_kernel_module(first_line):
            return "! kernels" == first_line.lower()

        self.kernels_dict = {}

        for file_path in self.source_files.iterate_paths(is_kernel_module):
            kernels_in_file = self.find_kernels_in_file(file_path)
            self.kernels_dict.update(kernels_in_file)

        return self.kernels_dict

    def find_kernels_in_file(self, file_path) -> dict[str, KernelFunctionDefinition]:
        reader = FortranFileReader(file_path, ignore_comments=False)
        f2008_parser = ParserFactory().create(std="f2008")
        parse_tree = FparserTree(f2008_parser(reader))

        subroutines = parse_tree.get_all_nodes_of_type("Subroutine_Subprogram")

        def is_kernel_subroutine(subroutine_node):
            comments = FparserTree(subroutine_node).get_all_nodes_in_children_of_type("Comment")

            return any(
                str(comment_node).lower().strip() == "! kernel"
                for comment_node in comments
            )

        kernel_functions = [x for x in subroutines if is_kernel_subroutine(x)]
        kernel_functions = [KernelFunctionDefinition(kernel_node) for kernel_node in kernel_functions]

        symbol_table = {kernel_function.name(): kernel_function for kernel_function in kernel_functions}

        for kernel_function in kernel_functions:
            kernel_function.set_symbol_table(symbol_table)

        return symbol_table

    def get_entry_kernels(self) -> list[KernelFunctionDefinition]:
        pass
