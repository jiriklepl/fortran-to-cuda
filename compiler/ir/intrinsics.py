"""Target-independent signatures and model inquiries for supported intrinsics."""

from dataclasses import dataclass

from .integers import INTEGER_MAX, constant_integer
from .nodes import CompilationError, Expr, Literal, ScalarType, SourceLocation


@dataclass(frozen=True)
class IntrinsicSignature:
    minimum_arguments: int
    maximum_arguments: int | None
    types: frozenset[ScalarType]


_NUMERIC = frozenset((ScalarType.INTEGER, ScalarType.REAL32, ScalarType.REAL))
_REAL = frozenset((ScalarType.REAL32, ScalarType.REAL))
REAL_MATH = frozenset(
    ("sqrt", "exp", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh")
)
KIND_ARGUMENT = frozenset(("real", "int", "nint", "floor", "ceiling"))
MODEL_INQUIRIES = frozenset(("kind", "epsilon", "tiny", "huge"))
ARRAY_INQUIRIES = frozenset(("size", "lbound", "ubound"))
INTRINSICS = {
    "abs": IntrinsicSignature(1, 1, _NUMERIC),
    "min": IntrinsicSignature(2, None, _NUMERIC),
    "max": IntrinsicSignature(2, None, _NUMERIC),
    **{name: IntrinsicSignature(1, 1, _REAL) for name in REAL_MATH},
    "atan2": IntrinsicSignature(2, 2, _REAL),
    **{name: IntrinsicSignature(2, 2, _NUMERIC) for name in ("mod", "modulo", "sign", "dim")},
    **{name: IntrinsicSignature(1, 2, _NUMERIC) for name in ("real", "int")},
    **{name: IntrinsicSignature(1, 2, _REAL) for name in ("nint", "floor", "ceiling")},
    "dble": IntrinsicSignature(1, 1, _NUMERIC),
    "merge": IntrinsicSignature(3, 3, frozenset(ScalarType)),
}

# Source keyword order; MIN/MAX additionally accept A1, A2, ... .
INTRINSIC_ARGUMENTS = {
    **{name: ("x",) for name in REAL_MATH | MODEL_INQUIRIES},
    "atan2": ("y", "x"),
    "abs": ("a",),
    "min": ("a1", "a2"),
    "max": ("a1", "a2"),
    **{name: ("a", "p") for name in ("mod", "modulo")},
    **{name: ("a", "b") for name in ("sign", "dim")},
    **{name: ("a", "kind") for name in KIND_ARGUMENT},
    "dble": ("a",),
    "merge": ("tsource", "fsource", "mask"),
    **{name: ("array", "dim", "kind") for name in ARRAY_INQUIRIES},
}


def intrinsic_kind(name: str, arguments: tuple[Expr, ...], location: SourceLocation) -> int | None:
    if name.lower() not in KIND_ARGUMENT or len(arguments) != 2:
        return None
    kind = constant_integer(arguments[1], location)
    if kind is None:
        raise CompilationError(f"{name.upper()} KIND must be a constant INTEGER expression", location)
    return kind


def intrinsic_type(
    name: str, types: tuple[ScalarType, ...], location: SourceLocation, *, kind: int | None = None
) -> ScalarType:
    name = name.lower()
    signature = INTRINSICS.get(name)
    if signature is None:
        raise CompilationError(f"unsupported intrinsic {name}", location)
    if len(types) < signature.minimum_arguments or (
        signature.maximum_arguments is not None and len(types) > signature.maximum_arguments
    ):
        raise CompilationError(f"wrong number of arguments for intrinsic {name.upper()}", location)
    if name == "merge":
        valid = types[0] == types[1] and types[2] is ScalarType.LOGICAL
    elif name in KIND_ARGUMENT:
        valid = types[0] in signature.types and (len(types) == 1 or types[1] is ScalarType.INTEGER)
    else:
        valid = len(set(types)) == 1 and types[0] in signature.types
    if not valid:
        raise CompilationError(f"invalid argument types for intrinsic {name.upper()}", location)
    if name in KIND_ARGUMENT:
        if len(types) == 2 and kind is None:
            raise CompilationError(f"{name.upper()} KIND must be a constant INTEGER expression", location)
        kind = 4 if kind is None else kind
        supported = {4, 8} if name == "real" else {4}
        if kind not in supported:
            raise CompilationError(f"unsupported {name.upper()} result kind {kind}", location)
        if name == "real":
            return ScalarType.REAL32 if kind == 4 else ScalarType.REAL
        return ScalarType.INTEGER
    if name == "dble":
        return ScalarType.REAL
    return types[0]


def model_inquiry(name: str, dtype: ScalarType, location: SourceLocation) -> Literal:
    """Inquiries depend on the declared type, never on the argument's value."""
    if name == "kind":
        return Literal("8" if dtype is ScalarType.REAL else "4", ScalarType.INTEGER)
    if name == "huge" and dtype is ScalarType.INTEGER:
        return Literal(str(INTEGER_MAX), dtype)
    if dtype not in _REAL:
        raise CompilationError(f"invalid argument types for intrinsic {name.upper()}", location)
    binary32 = dtype is ScalarType.REAL32
    values = {
        "epsilon": 2.0 ** (-23 if binary32 else -52),
        "tiny": 2.0 ** (-126 if binary32 else -1022),
        "huge": float.fromhex("0x1.fffffep+127" if binary32 else "0x1.fffffffffffffp+1023"),
    }
    return Literal(repr(values[name]), dtype)
