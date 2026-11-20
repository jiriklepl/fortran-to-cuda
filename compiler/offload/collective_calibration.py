"""Offline measurements of source-proven, generated persistent-team workers.

The fixture uses the public source-scope artifacts. It does not reproduce the
coordinator algorithm, measure an application, or tune a live invocation.
"""

from __future__ import annotations

import shlex


def normalize_fortran_options(raw: str) -> str:
    """Remove only build-location/output options; preserve ordered semantics."""
    result, skip = [], False
    for token in shlex.split(raw):
        if skip:
            skip = False
        elif token in {"-I", "-J", "-o"}:
            skip = True
        elif token.startswith(("-openmp", "-offload", "-opt")) or not any(
            token.startswith(flag) and len(token) > len(flag) for flag in ("-I", "-J", "-o")
        ):
            result.append(token)
    if skip:
        raise ValueError("Fortran location/output flag is missing its argument")
    return "\x1f".join(result)




