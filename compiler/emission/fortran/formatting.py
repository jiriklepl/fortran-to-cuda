"""Free-form line wrapping shared by ordinary and workspace bridges."""


def _fortran_list(prefix: str, values: list[str], suffix: str, indentation: int) -> list[str]:
    """Continue every argument separately to stay within free-form line limits."""
    if not values:
        return [" " * indentation + prefix + suffix]
    lines = [" " * indentation + prefix + " &"]
    lines.extend(
        " " * (indentation + 4) + value + (", &" if index < len(values) - 1 else " &" if suffix else "")
        for index, value in enumerate(values)
    )
    if suffix:
        lines.append(" " * indentation + suffix)
    return lines


