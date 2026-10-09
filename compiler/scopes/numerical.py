"""Public source-extraction packages for independently normalized procedures.

Extraction owns normalization/capture/OpenMP facts. The compiler checks their
source identity, bindings and coordinate contract, then proves numerical legality
using its ordinary frontend. Original procedure definition events stay in clones.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects, _kind
from compiler.ir import CompilationError, SourceLocation


@dataclass(frozen=True)
class Parameter:
    name: str
    resource: str
    rank: int
    lower_bound_dimension: int | None = None
    runtime_lower_bound: bool = False


@dataclass(frozen=True)
class NumericalSource:
    procedure: str
    path: Path
    entry: str
    parameters: tuple[Parameter, ...]
    digest: str

    @property
    def arrays(self):
        return {p.resource for p in self.parameters if p.rank}

    @property
    def runtime_origins(self):
        """Original allocation descriptors required even without payload access."""
        return {p.resource for p in self.parameters if p.runtime_lower_bound}


def resource_binding(analysis, routine, resource):
    """Resolve a source resource through actual visible bindings, including aliases."""
    if not isinstance(resource, str):
        raise CompilationError("numerical source parameter needs a canonical resource")
    if resource.startswith("argument::"):
        name = resource.split("::", 1)[1]
        binding = routine.scope.bindings.get(name)
        if name not in routine.arguments or binding is None:
            raise CompilationError("numerical source resource is not an original argument: " + resource)
        return binding
    candidates = {resource.rsplit("::", 1)[-1]}
    visited = set()

    def imported(module):
        target = analysis.modules.get(module)
        if target is None or module in visited:
            return
        visited.add(module)
        candidates.update(target.bindings)
        candidates.update(target.imports)
        for name in target.wildcards:
            imported(name)

    current = routine.scope
    while current:
        candidates.update(current.bindings)
        candidates.update(current.imports)
        for module in current.wildcards:
            imported(module)
        current = current.parent
    for name in sorted(candidates):
        binding = analysis._binding(routine.scope, name)
        if binding and binding.root == resource:
            return binding
    raise CompilationError("numerical source resource is unavailable in original procedure: " + resource)


def load_numerical_sources(document, analysis):
    if document is None:
        return {}
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or document.get("source_inputs") != analysis.sources or not isinstance(document.get("entries"), list)):
        raise CompilationError("numerical source package requires versioned matching source inputs")
    if len(document["entries"]) > analysis.procedure_limit:
        raise CompilationError("numerical source package exceeds the bounded procedure budget")
    result = {}
    for item in document["entries"]:
        if not isinstance(item, dict):
            raise CompilationError("numerical source entries must be objects")
        procedure = item.get("procedure")
        routine = analysis.routines.get(procedure)
        if routine is None or procedure in result:
            raise CompilationError("numerical source procedure is unavailable or duplicated")
        if (item.get("normalization") != "whole_storage_rebased_v1"
                or item.get("participation") != "serial_coordinator"
                or item.get("capture_safe") is not True or item.get("preserves_source_order") is not True
                or item.get("source_sha256") != analysis.sources[str(routine.scope.path)]):
            raise CompilationError("numerical source package is missing source/capture/coordinate facts")
        try:
            path = Path(item["path"]).resolve(strict=True)
            digest = sha256(path.read_bytes()).hexdigest()
        except (KeyError, TypeError, OSError) as error:
            raise CompilationError("numerical source path is unavailable") from error
        if digest != item.get("sha256"):
            raise CompilationError("normalized numerical source hash differs from its package")
        normalized = SourceEffects([path])
        entry = item.get("entry")
        target = normalized.routines.get(entry)
        if target is None or not isinstance(item.get("parameters"), list):
            raise CompilationError("numerical source package lacks a qualified entry/parameter mapping")
        supplied = item["parameters"]
        if len(supplied) != len(target.arguments):
            raise CompilationError("numerical source parameter mapping differs from its signature")
        parameters = []
        access_intents = {}
        dynamic_bounds = {}
        for formal, mapping in zip(target.arguments, supplied, strict=True):
            if not isinstance(mapping, dict) or mapping.get("name") != formal:
                raise CompilationError("numerical source mapping must preserve parameter order")
            binding = resource_binding(analysis, routine, mapping.get("resource"))
            expected = target.scope.bindings[formal]
            dimension = mapping.get("lower_bound_dimension")
            dynamic = binding.root in analysis.stable_module_allocatables
            if dimension is not None:
                if (type(dimension) is not int or not 1 <= dimension <= binding.rank
                        or expected.signature() != ("integer", 4, 0) or expected.intent != "in"):
                    raise CompilationError("numerical lower-bound parameter has an invalid type/dimension")
                if dynamic:
                    dimensions = dynamic_bounds.setdefault(binding.root, set())
                    if dimension in dimensions:
                        raise CompilationError("numerical allocation origins require unique lower-bound dimension mappings")
                    dimensions.add(dimension)
                else:
                    lower = routine.scope.kinds.integer(F.Level_2_Expr(binding.lower_bounds[dimension-1]),
                                                        SourceLocation(str(routine.scope.path)))
                    if not -(2**31) <= lower < 2**31:
                        raise CompilationError("numerical source lower bound exceeds the supported INTEGER ABI")
                parameters.append(Parameter(formal, binding.root, 0, dimension, dynamic))
                continue
            if expected.signature() != binding.signature():
                raise CompilationError("numerical source resource type/kind/rank differs: " + formal)
            if expected.dtype == "logical" and expected.kind != 1:
                raise CompilationError("numerical source LOGICAL conversion requires guarded ABI handling")
            if expected.intent not in ({"in", "inout"} if expected.rank else {"in"}):
                raise CompilationError("numerical source access intents must be in/inout; definition events belong to original source")
            if expected.rank:
                if mapping.get("physical_origin") != [0]*binding.rank or any(v != "1" for v in expected.lower_bounds):
                    raise CompilationError("numerical source arrays require a whole-storage zero-origin mapping")
                if binding.intent == "in" and expected.intent != "in":
                    raise CompilationError("numerical source writes an original INTENT(IN) argument")
                if dynamic:
                    dynamic_bounds.setdefault(binding.root, set())
            if binding.root in access_intents:
                raise CompilationError("numerical source maps duplicate resource arguments")
            access_intents[binding.root] = expected.intent
            parameters.append(Parameter(formal, binding.root, binding.rank))
        for resource, dimensions in dynamic_bounds.items():
            binding = resource_binding(analysis, routine, resource)
            if dimensions != set(range(1, binding.rank + 1)):
                raise CompilationError("numerical allocation origins require every original runtime lower-bound dimension: "
                                       + resource)
        # Calls with source entry effects must remain separate context-aware
        # workers. Numerical helper inlining cannot silently remove those events.
        if any(_kind(node) == "Call_Stmt" for node in walk(routine.execution)):
            raise CompilationError("numerical source packages initially require original leaf procedures")
        for operation in analysis.summarize(procedure)["operations"]:
            if operation["kind"] in {"write", "overwrite"}:
                if not operation["rank"]:
                    raise CompilationError("numerical source packages require read-only external scalars")
                if access_intents.get(operation["resource"]) != "inout":
                    raise CompilationError("numerical source mapping omits an original written resource")
        result[procedure] = NumericalSource(procedure, path, entry, tuple(parameters), digest)
    return result
