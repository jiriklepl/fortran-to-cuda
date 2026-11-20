"""Bounded native sections from original ASTs, separate from whole effects.

Assignments and bounded affine counted loops are refined. The original native
computation is retained; these facts describe its physical array accesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compiler.frontend.call_bindings import SourceDependency


@dataclass(frozen=True)
class NativeBound:
    kind: str
    value: int | None = None
    dimension: int | None = None
    resource: str | None = None
    operator: str | None = None
    children: tuple[NativeBound, ...] = ()
    dependencies: tuple[SourceDependency, ...] = field(default=(), compare=False, repr=False)

    def public(self):
        return {"kind": self.kind, **({"value": self.value} if self.value is not None else {}),
                **({"dimension": self.dimension} if self.dimension is not None else {}),
                **({"resource": self.resource} if self.resource is not None
                   and self.kind in {"scalar", "lbound", "ubound", "size"} else {}),
                **({"operator": self.operator} if self.operator is not None else {}),
                **({"children": [child.public() for child in self.children]} if self.children else {})}


@dataclass(frozen=True)
class NativeIteration:
    lower: NativeBound
    upper: NativeBound
    step: int = 1

    def public(self):
        return {"lower": self.lower.public(), "upper": self.upper.public(), "step": self.step}




















