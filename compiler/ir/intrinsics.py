"""Target-independent signatures for the supported scalar intrinsic functions."""

from dataclasses import dataclass

from .nodes import CompilationError, ScalarType, SourceLocation


@dataclass(frozen=True)
class IntrinsicSignature:
    minimum_arguments: int
    maximum_arguments: int | None
    types: frozenset[ScalarType]


_NUMERIC = frozenset((ScalarType.INTEGER, ScalarType.REAL32, ScalarType.REAL))
INTRINSICS = {
    "abs": IntrinsicSignature(1, 1, _NUMERIC),
    "min": IntrinsicSignature(2, None, _NUMERIC),
    "max": IntrinsicSignature(2, None, _NUMERIC),
    "sqrt": IntrinsicSignature(1, 1, frozenset((ScalarType.REAL32, ScalarType.REAL))),
}


def intrinsic_type(name: str, types: tuple[ScalarType, ...], location: SourceLocation) -> ScalarType:
    signature = INTRINSICS.get(name.lower())
    if signature is None:
        raise CompilationError(f"unsupported intrinsic {name}", location)
    if len(types) < signature.minimum_arguments or (
        signature.maximum_arguments is not None and len(types) > signature.maximum_arguments
    ):
        raise CompilationError(f"wrong number of arguments for intrinsic {name.upper()}", location)
    if len(set(types)) != 1 or types[0] not in signature.types:
        raise CompilationError(f"invalid argument types for intrinsic {name.upper()}", location)
    return types[0]
