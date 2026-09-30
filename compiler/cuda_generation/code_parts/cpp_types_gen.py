from __future__ import annotations

from compiler.context import Variable
from compiler.typing import TerminalType, Type


class CppTyper:
    TYPE_TO_CPP_TYPE_STR = {
        "integer": "int",
        "real(kind = knd)": "double"
    }

    TYPE_TO_FORTRAN_TYPE_STR = {
        "integer": "integer",
        "real(kind = knd)": "real(kind = knd)"
    }

    def get_cpp_type_str(self, variable: Type) -> str:
        base_type = variable.get_underlying_type()
        result = self.TYPE_TO_CPP_TYPE_STR[base_type.name.lower()]
        
        if variable.is_array():
            result += "*"
    
        return result
    
    def size_of(self, type: Type) -> str:
        type_str = self.TYPE_TO_CPP_TYPE_STR[type.name.lower()]  
        return f'sizeof({type_str})'

    def get_size_t(self) -> str:
        return "size_t"


class FortranTyper:
    TYPE_TO_FORTRAN_TYPE_STR = {
        "integer": "integer(c_int)",
        "real(kind = knd)": "real(kind = knd)"
    }

    def __init__(self, variable_namer: "VariableNamer"):
        self.namer = variable_namer

    def get_fortran_type_for_cpp_bind(self, variable: Type) -> str:
        base_type = variable.get_underlying_type().name.lower()

        if variable.is_array():
            return f"{base_type}, dimension(*)"

        return self.TYPE_TO_FORTRAN_TYPE_STR[base_type] + ", value, intent(in)"
    
    def get_size_t_cpp_bind(self) -> str:
        return "integer(c_size_t), value, intent(in)"
    
    def cast_to_cpp_bind(self, var: Variable) -> str:
        type = var.type()

        var_name = self.namer.format_name(var)

        if type.is_array():
            return var_name

        if type.name.lower() == "integer":
            return f"int({var_name}, kind=c_int)"

        if type.name.lower() == "real(kind = knd)":
            return f"real({var_name}, kind=c_double)"

        raise NotImplementedError(f"Unsupported type for Fortran to C++ bind: {type}")

    def casted_size(self, var: Variable, dim_num) -> str:
        var_name = self.namer.format_name(var)
        return f"int(size({var_name}, {dim_num + 1}), kind=c_size_t)"
    
    def get_fortran_type_decl(self, variable: Variable) -> str:
        type = variable.type()
        base_type = type.get_underlying_type().name.lower()

        if variable.is_input_output():
            intent = ", intent(inout)"
        elif variable.is_input():
            intent = ", intent(in)"
        elif variable.is_output():
            intent = ", intent(out)"
        else:
            intent = ""

        if type.is_array():
            return f"{base_type}, dimension({','.join([':'] * type.get_dim_count())})" + intent

        return self.TYPE_TO_FORTRAN_TYPE_STR[base_type] + intent