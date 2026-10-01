from compiler.debugging.color_printer import Colors as c


class Type:
    def get_dim_count(self) -> int:
        raise NotImplementedError()

    def is_base_type(self):
        raise NotImplementedError()

    def is_array(self):
        return self.get_dim_count() != 0

    def get_underlying_type(self):
        raise NotImplementedError()


class TerminalType(Type):
    def __init__(self, name):
        self.name = name

    def get_dim_count(self) -> int:
        return 0

    def is_base_type(self):
        return True

    def get_underlying_type(self):
        return self

    def __str__(self):
        return f"{c.TYPE}T_{self.name}{c.END}"


class ArrayType(Type):
    def __init__(self, element_type: Type, spec_ast):
        self.element_type = element_type
        self.spec_ast = spec_ast

    def get_dim_count(self) -> int:
        return 1 + self.element_type.get_dim_count()

    def is_base_type(self):
        return False

    def get_underlying_type(self):
        return self.element_type.get_underlying_type()

    def __str__(self):
        return f"{str(self.element_type)}{c.TYPE}[]{c.END}"
