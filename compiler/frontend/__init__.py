"""Fortran parsing and lowering into the computation IR."""

from .lowering import lower_file

__all__ = ["lower_file"]
