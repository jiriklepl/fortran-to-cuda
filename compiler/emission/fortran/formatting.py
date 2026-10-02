"""Free-form line wrapping shared by ordinary and workspace bridges."""


def _fortran_list(prefix: str, values: list[str], suffix: str, indentation: int) -> list[str]:
    """Continue every argument separately to stay within free-form line limits."""
    if not values:
        return [" " * indentation + prefix + suffix]
    lines = [" " * indentation + prefix + " &"]
    lines.extend(
        " " * (indentation + 4) + value + (", &" if index < len(values) - 1 else " &")
        for index, value in enumerate(values)
    )
    lines.append(" " * indentation + suffix)
    return lines


def _fortran_line(declaration: str, indentation: int) -> list[str]:
    line = " " * indentation + declaration
    if len(line) <= 132:
        return [line]
    # A rank-15 assumed shape and a long public dummy name can exceed the
    # free-form limit. A continuation before its name keeps both parts short.
    left, right = line.split("::", 1)
    return [left.rstrip() + " :: &", " " * (indentation + 4) + right.strip()]
