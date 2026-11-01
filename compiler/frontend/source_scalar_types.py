"""Types of unchanged original scalar numeric operations; never their values."""

SOURCE_SCALAR_OPERATOR_VERSION = 1

_SUPPORTED = frozenset({("integer", 4, 0), ("integer", 8, 0),
                        ("real", 4, 0), ("real", 8, 0)})


def scalar_numeric_result(operator, left, right):
    """Return the supported target ABI signature or decline the operation.

    Fortran 2023 10.1.5.1 and 10.1.9.3 determine type and kind independently
    of evaluation. In particular REAL ** INTEGER keeps the REAL kind, while
    two INTEGER kinds use the greater exponent range, including for **.
    This service never folds, rewrites or authorizes evaluating an expression.
    """
    if operator not in {"+", "-", "*", "/", "**"} or left not in _SUPPORTED or right not in _SUPPORTED:
        return None
    reals = [value[1] for value in (left, right) if value[0] == "real"]
    return ("real", max(reals), 0) if reals else ("integer", max(left[1], right[1]), 0)
