from __future__ import annotations

from compiler.debugging.color_printer import Colors as c
from compiler.fparser_tree_abstraction import FparserTree, LoopStatement
from compiler.typing import ArrayType, TerminalType


class Variable:
    def __init__(self, name, type, attributes: str | list[str] = ""):
        self._name = name
        self._type = type
        self.attributes: list[str] = attributes.split(",") if isinstance(attributes, str) else attributes
        self.attributes = [attr.strip()
                           for attr_item in self.attributes
                           for attr in attr_item.split(",")]

        self._is_param = False
        self._suffix = ""

    def is_input(self):
        return "intent(in)" in self.attributes or "intent(inout)" in self.attributes

    def is_output(self):
        return "intent(out)" in self.attributes or "intent(inout)" in self.attributes
    
    def type(self) -> TerminalType | ArrayType:
        return self._type

    def is_input_output(self):
        return "intent(inout)" in self.attributes
    
    def is_function_param(self):
        return self._is_param
    
    def is_iterator_var(self):
        return False

    def set_as_param(self):
        self._is_param = True

    def set_name_suffix(self, suffix):
        self._suffix = suffix

    def name(self):
        return self._name + self._suffix

    def __str__(self):
        return f"{c.CLASS}Variable{c.END}({c.FIELD}name{c.END}={c.VAR}{self.name()}{c.END}, {c.FIELD}type{c.END}={self.type()}, {c.FIELD}attributes{c.END}={c.ATTR}{self.attributes}{c.END})"
    
    @staticmethod
    def from_(intrinsic_type_ast, variable_ast, attributes):
        base_type_name = str(intrinsic_type_ast[0]).lower()
        base_type = TerminalType(base_type_name)

        variable_ast = FparserTree(variable_ast)
        variable_name = str(variable_ast.get_all_nodes_in_children_of_type("Name")[0])

        def is_array_dim_spec(node):
            return node.__class__.__name__.lower().endswith("_Shape_Spec".lower())

        shape_list = variable_ast.get_nodes(is_array_dim_spec)

        final_type = base_type

        for shape in shape_list:
            final_type = ArrayType(final_type, shape)

        return Variable(variable_name, final_type, attributes)


class IterationVariable:
    def __init__(self, original_variable: Variable, loop_statement: LoopStatement):
        self.original_variable = original_variable
        self.loop_statement = loop_statement

    def name(self):
        return self.original_variable.name()

    def is_named(self, name):
        return self.name() == name
    
    def is_function_param(self):
        return False
    
    def type(self):
        return self.original_variable.type()
    
    def is_iterator_var(self):
        return True
    
    def __str__(self):
        return f"{c.CLASS}IterationVariable{c.END}({c.FIELD}name{c.END}={c.VAR}{self.name()}{c.END}, {c.FIELD}original_variable{c.END}={self.original_variable})"


class Context:
    def _load_variables(self, specifications_ast, declaration_ast):
        tree = FparserTree(specifications_ast)
        specification_statements = tree.get_all_nodes_of_type("Type_Declaration_Stmt")

        all_variables = []
        for specification in specification_statements:
            variables = self._load_variables_from_specification(specification)
            all_variables.extend(variables)

        dummy_arg_list = FparserTree(declaration_ast).get_all_nodes_of_type("Dummy_Arg_List")
        dummy_arg_list = dummy_arg_list[0].children if dummy_arg_list else []
        dummy_names = [str(arg) for arg in dummy_arg_list]

        for variable in all_variables:
            if variable.name() in dummy_names:
                variable.set_as_param()

        return all_variables
    
    def _load_variables_from_specification(self, specification_ast):
        tree = FparserTree(specification_ast)

        intrinsic_type = tree.get_all_nodes_of_type("Intrinsic_Type_Spec")
        variable_list = tree.get_all_nodes_of_type("Entity_Decl")
        attributes = [str(attr).lower() for attr in tree.get_all_nodes_of_type("Attr_Spec_List")]

        variables = []
        for var in variable_list:
            variables.append(Variable.from_(intrinsic_type, var, attributes))

        return variables
    
    def get_variable_by_name(self, name) -> Variable:
        raise NotImplementedError("This method should be implemented by subclasses of Context")

    def variable_defined(self, variable: Variable) -> bool:
        raise NotImplementedError("This method should be implemented by subclasses of Context")

    def enum_do_loop_contexts(self):
        raise NotImplementedError("This method should be implemented by subclasses of Context")

    def is_call_context(self):
        return False
    
class LocalContext(Context):
    def __init__(self, specifications_ast, declaration_ast):
        self.specifications_ast = FparserTree(specifications_ast)
        self.declaration_ast = FparserTree(declaration_ast)
        self.variables = self._load_variables(self.specifications_ast, self.declaration_ast)

    def get_variable_by_name(self, name) -> Variable:
        for variable in self.variables:
            if variable.name() == name:
                return variable
        
        raise Exception(f"Variable with name {name} not found in context")

    def variable_defined(self, variable: Variable) -> bool:
        for var in self.variables:
            if var.name() == variable.name():
                return True
        
        return False

    def __str__(self):
        variables_str = "\n".join([str(var) for var in self.variables])
        tabbed_variables_str = "\t" + variables_str.replace("\n", "\n\t")
        return f"{c.CLASS}LocalContext{c.END}({c.FIELD}variables{c.END}=\n{tabbed_variables_str}\n)"
    
    def get_call_arg_names(self):
        arg_list = self.declaration_ast.get_all_nodes_of_type("Dummy_Arg_List")
        arg_list = FparserTree(arg_list[0]).children()
        return [str(arg) for arg in arg_list]
    
    def enum_do_loop_contexts(self):
        return [] 


class DoLoopContext(Context):
    def __init__(self, do_statement: LoopStatement, parent_context: Context):
        self.parent_context = parent_context
        self.loop_statement = do_statement

        iter_var_name = do_statement.iteration_variable_name()
        self.iteration_variable = IterationVariable(
            self.parent_context.get_variable_by_name(iter_var_name),
            do_statement)

    def get_iteration_variable(self) -> IterationVariable:
        return self.iteration_variable

    def get_variable_by_name(self, name) -> Variable | IterationVariable:
        if self.iteration_variable.is_named(name):
            return self.iteration_variable
        
        return self.parent_context.get_variable_by_name(name)

    def variable_defined(self, variable: Variable) -> bool:
        if self.iteration_variable.is_named(variable.name()):
            return True
        
        return self.parent_context.variable_defined(variable)

    def enum_do_loop_contexts(self):
        yield self
        yield from self.parent_context.enum_do_loop_contexts()

    def enum_range_code(self):
        loop_control_part = self.loop_statement.get_loop_control_part()
        range_from_to = loop_control_part.children[1]
        range_from = range_from_to[0]
        range_to = range_from_to[1]
        
        yield range_from, self
        yield range_to, self

    def range_code_ast_s(self) -> tuple[any, any, any]:
        return self.loop_statement.range_code_ast_s()

    def __str__(self):
        parent_context_str = str(self.parent_context)
        tabbed_parent_context_str = "\t" + parent_context_str.replace("\n", "\n\t")
        return f"{c.CLASS}DoLoopContext{c.END}({c.FIELD}iteration_variable{c.END}={self.iteration_variable}, {c.FIELD}parent_context{c.END}=\n{tabbed_parent_context_str}\n)"


class ContextWithArguments(Context):
    def __init__(self,
                 function_local_context: LocalContext,
                 caller_context: Context,
                 call_arguments: list[Variable]):
        self.function_local_context = function_local_context
        self.call_arguments = call_arguments
        self.caller_context = caller_context

        function_arg_list = function_local_context.get_call_arg_names()
        if len(function_arg_list) != len(call_arguments):
            raise Exception(f"Number of call arguments ({len(call_arguments)}) does not match number of function arguments ({len(function_arg_list)})")
        
        self.translation_dict = {arg_name: arg for arg_name, arg in zip(function_arg_list, call_arguments)}
        self.translation_dict.update(self._rename_reused_variables())

    def get_variable_by_name(self, name) -> Variable:
        if name in self.translation_dict:
            return self.translation_dict[name]

        raise Exception(f"Variable with name {name} not found in context")

    def variable_defined(self, variable: Variable) -> bool:
        if variable.name() in self.translation_dict:
            return True

        return self.function_local_context.variable_defined(variable)

    def _rename_reused_variables(self) -> dict[str, Variable]:
        renamed_dict = {}

        for local_var in self.function_local_context.variables:

            if local_var.name() in self.translation_dict:
                continue

            original_name = local_var.name()

            i = 0
            while self.caller_context.variable_defined(local_var):
                local_var.set_name_suffix(f"_{i}")
                i += 1

            renamed_dict[original_name] = local_var

        return renamed_dict

    def is_call_context(self):
        return True
    
    def enum_do_loop_contexts(self):
        yield from self.caller_context.enum_do_loop_contexts()

    def __str__(self):
        function_local_context_str = str(self.function_local_context)
        tabbed_function_local_context_str = "\t" + function_local_context_str.replace("\n", "\n\t")

        caller_context_str = str(self.caller_context)
        tabbed_caller_context_str = "\t" + caller_context_str.replace("\n", "\n\t")

        arguments_str = "\n".join([str(arg) for arg in self.call_arguments])
        tabbed_arguments_str = "\t" + arguments_str.replace("\n", "\n\t")

        return f"{c.CLASS}ContextWithArguments{c.END}({c.FIELD}call_arguments{c.END}=\n{tabbed_arguments_str}, \n\t{c.FIELD}function_local_context{c.END}={tabbed_function_local_context_str}, \n{c.FIELD}caller_context{c.END}=\n{tabbed_caller_context_str}\n)"
    