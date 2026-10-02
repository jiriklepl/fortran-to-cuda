"""Structured parallel-proof outcomes. Semantic errors never become fallback."""

from dataclasses import dataclass

from compiler.ir import CompilationError, ParallelRegion, SourceLocation


@dataclass(frozen=True)
class ProofFailure:
    reason: str
    location: SourceLocation
    witness: str | None = None
    conservative: bool = False


class ParallelizationError(CompilationError):
    def __init__(self, reason: str, location: SourceLocation, *, witness=None, conservative=False):
        self.failure = ProofFailure(reason, location, witness, conservative)
        super().__init__(reason, location)


@dataclass(frozen=True)
class ParallelProof:
    region: ParallelRegion | None = None
    failure: ProofFailure | None = None

    @property
    def proven(self) -> bool:
        return self.region is not None
