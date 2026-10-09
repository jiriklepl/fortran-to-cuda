"""Bound compiler-generated wrappers without specializing runtime placement."""

from dataclasses import dataclass
from hashlib import sha256

from compiler.ir import CompilationError


@dataclass(frozen=True)
class Variant:
    procedure: str
    interface: str
    role: str
    name: str
    summary_identity: str
    requirements: tuple[str, ...]
    shared_artifacts: tuple[str, ...]

    @property
    def identity(self):
        value = "\0".join((self.procedure, self.summary_identity, self.role, self.interface))
        return sha256(value.encode()).hexdigest()

    def public(self):
        return {"identity": self.identity, "procedure": self.procedure, "role": self.role,
                "interface": self.interface, "name": self.name, "summary_identity": self.summary_identity,
                "placement": "runtime mode; no shape or CPU/GPU partition specialization",
                "requirements": list(self.requirements), "shared_artifacts": list(self.shared_artifacts)}


class VariantRegistry:
    """A rejected source candidate can restore its complete generation budget."""

    def __init__(self, *, per_procedure=4, total=128):
        if type(per_procedure) is not int or type(total) is not int or min(per_procedure, total) < 1:
            raise ValueError("variant limits must be positive integers")
        self.per_procedure, self.total = per_procedure, total
        self._variants = {}

    def register(self, procedure, *, interface, role, name, summary_identity, requirements=(), shared_artifacts=()):
        variant = Variant(procedure, interface, role, name, summary_identity,
                          tuple(requirements), tuple(shared_artifacts))
        previous = self._variants.get(variant.identity)
        if previous is not None:
            if previous != variant:
                raise CompilationError("generated source variant identity changed within one compilation: " + procedure)
            return previous
        if sum(item.procedure == procedure for item in self._variants.values()) >= self.per_procedure:
            raise CompilationError("generated wrapper variant budget exhausted for " + procedure
                                   + ": limit " + str(self.per_procedure))
        if len(self._variants) >= self.total:
            raise CompilationError("generated wrapper variant compilation budget exhausted: limit " + str(self.total))
        self._variants[variant.identity] = variant
        return variant

    def checkpoint(self):
        return dict(self._variants)

    def restore(self, checkpoint):
        self._variants = dict(checkpoint)

    def public(self):
        procedures = sorted({item.procedure for item in self._variants.values()})
        return {"schema_version": 1, "limits": {"per_procedure": self.per_procedure, "compilation": self.total},
                "generated_count": len(self._variants), "original_native_entries": "retained without generation",
                "procedures": [{"procedure": procedure, "native_entry": procedure,
                                "variants": [item.public() for item in self._variants.values()
                                             if item.procedure == procedure]}
                               for procedure in procedures]}
