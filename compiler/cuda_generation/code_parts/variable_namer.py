
from __future__ import annotations

from compiler.cuda_generation.code_parts.cpp_types_gen import CppTyper
from compiler.debugging.color_printer import Colors as c
from compiler.context import Context, Variable


class VariableNamer:
    def __init__(self):
        self.cpp_typer = CppTyper()

    def get_name(self, node, context: Context) -> str:
        if node.__class__.__name__.lower() == "name":
            return self.get_name_from_name_node(node, context)
        else:
            raise Exception(f"Unsupported node type {c.CLASS}{node.__class__.__name__}{c.END} for naming. Function at: {__file__}")

    def get_name_from_name_node(self, name_node, context: Context) -> str:
        original_name_used_in_context = str(name_node)
        variable = context.get_variable_by_name(original_name_used_in_context)
        
        real_variable_name = self.format_name(variable)
        return real_variable_name
    
    def get_get_dim_sizes_variable_names_of(self, name_node, context: Context) -> list[str]:
        original_name_used_in_context = str(name_node)
        variable = context.get_variable_by_name(original_name_used_in_context)
        
        dim_count = variable.type().get_dim_count()
        variable_name = self.format_name(variable)

        return [self.format_array_dim_size_variable_name(variable_name, dim_num) 
                for dim_num in range(dim_count)]
    
    def format_array_dim_size_variable_name(self, array_variable: Variable | str, dim_num):
        if isinstance(array_variable, Variable):
            array_variable_name = self.format_name(array_variable)
        else:
            array_variable_name = array_variable

        return f"{array_variable_name}_dim{dim_num + 1}"
    
    def format_name(self, variable: Variable) -> str:
        return variable.name().lower()
    
    def format_device_name(self, variable: Variable) -> str:
        if not variable.type().is_array():
            return self.format_name(variable)
        
        return f"{self.format_name(variable)}_device"
    
    class IterNames:
        def __init__(self, variable: Variable, type_str):
            self.type = type_str
            self.name = f"{variable.name().lower()}"
            self.from_name = f"{variable.name().lower()}_from"
            self.to_name = f"{variable.name().lower()}_to"
            self.step_name = f"{variable.name().lower()}_step"

    def format_iter_var_names(self, variable: Variable) -> IterNames:
        type_str = self.cpp_typer.get_cpp_type_str(variable.type())
        return VariableNamer.IterNames(variable, type_str)