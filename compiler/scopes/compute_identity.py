"""Check calibrated native semantics in each original Fortran source object.

Generated numerical workers can live in another file with different options.
Their compiler intrinsics, or an outer caller's intrinsics, cannot authenticate
the original child counterfactual. These nonmutating helpers remain in the
original modules and run before numerical work; they establish no GPU legality.
"""

from dataclasses import dataclass
from hashlib import sha256

from fparser.two import Fortran2003 as F

from compiler.emission.fortran.formatting import _fortran_list
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError

_LITERAL_PART_LIMIT = 192


@dataclass(frozen=True)
class ComputeIdentityRequirement:
    source: str
    source_sha256: str
    module: str
    procedures: tuple[str, ...]
    helper: str
    compiler_version: str
    semantic_options: str
    runtime_schedule: dict | None = None

    def public(self):
        return {"source": self.source, "source_sha256": self.source_sha256,
                "module": self.module, "procedures": list(self.procedures), "helper": self.helper,
                "fortran": {"compiler_version": self.compiler_version,
                            "semantic_options": self.semantic_options},
                **({"runtime_schedule": self.runtime_schedule} if self.runtime_schedule is not None else {})}


@dataclass(frozen=True)
class ComputeIdentityGuard:
    available: bool = True
    required: bool = False
    imports: tuple[str, ...] = ()
    expression: str = ".true."
    requirements: tuple[ComputeIdentityRequirement, ...] = ()
    reason: str | None = None

    def public(self):
        return {"schema_version": 1, "available": self.available, "required": self.required,
                "reason": self.reason, "requirements": [item.public() for item in self.requirements],
                "execution": "nonmutating original-object compatibility checks before numerical work",
                "mismatch": "coherent native execution of the reached unit; never replay an executed prefix"}


def _unavailable(reason):
    return ComputeIdentityGuard(available=False, required=True, expression=".false.", reason=reason)


def _literal_parts(value, *, semantic=False):
    if not isinstance(value, str) or not value or any(ord(char) < 32 and char != "\x1f" for char in value):
        raise CompilationError("compute Fortran identity contains unsupported characters")
    if not semantic and "\x1f" in value:
        raise CompilationError("compute Fortran version contains a semantic separator")
    parts = []
    for token in value.split("\x1f"):
        if parts:
            parts.append("achar(31)")
        parts.extend("'" + token[index:index + 48].replace("'", "''") + "'"
                     for index in range(0, len(token), 48))
        if not token:
            parts.append("''")
    if len(parts) > _LITERAL_PART_LIMIT:
        raise CompilationError("compute Fortran identity exceeds bounded literal generation")
    return parts


def _helper(name, version, semantic, *, runtime_schedule=False):
    # ACHAR is local and explicitly intrinsic: a same-named original caller or
    # host binding cannot change interpretation of semantic token separators.
    version_parts, semantic_parts = _literal_parts(version), _literal_parts(semantic, semantic=True)
    if len(version_parts) + len(semantic_parts) > _LITERAL_PART_LIMIT:
        raise CompilationError("compute Fortran identity exceeds bounded literal generation")
    arguments = ["fort_version() // fort_nul", "fort_options() // fort_nul",
                 " // &\n        ".join((*version_parts, "fort_nul")),
                 " // &\n        ".join((*semantic_parts, "fort_nul"))]
    return "\n".join([
        "logical function " + name + "() result(fort_matches)",
        "use, intrinsic :: iso_fortran_env, only: fort_version => compiler_version, &",
        "    fort_options => compiler_options",
        "use, intrinsic :: iso_c_binding, only: fort_nul => c_null_char",
        "use fort_scoped_team_observer, only: fort_compatible => fort_scope_team_fortran_compatible_v1",
        *(["use omp_lib, only: fort_get_schedule => omp_get_schedule, &",
           "    fort_sched_kind => omp_sched_kind, fort_static => omp_sched_static"] if runtime_schedule else []),
        "implicit none", "intrinsic :: achar",
        *(["intrinsic :: iand, huge", "integer(fort_sched_kind) :: fort_schedule",
           "integer :: fort_chunk"] if runtime_schedule else []),
        *_fortran_list("fort_matches = fort_compatible(", arguments, ") /= 0", 0),
        *(["if (.not. fort_matches) return", "call fort_get_schedule(fort_schedule,fort_chunk)",
           "fort_matches = fort_chunk == 0 .and. iand(fort_schedule,huge(fort_schedule)) == fort_static"]
          if runtime_schedule else []),
        "end function " + name, ""])


def _module(analysis, routine):
    scope = routine.scope
    while scope.parent is not None:
        scope = scope.parent
    if analysis.modules.get(scope.module) is not scope or _kind(scope.node) != "Module":
        raise CompilationError("compute identity requires an original module source object")
    return scope


def _conflicts(analysis, scope, name):
    return (analysis._binding(scope, name) is not None
            or bool(analysis._candidates(scope, F.Name(name)))
            or name in scope.imports or name in scope.ambiguous_imports or name in scope.generics)


def compute_identity_guard(builder, procedures, *, caller=None):
    """Prepare only guards for referenced, already-generated compute models.

    The caller applies ``imports`` at its original lexical specification and
    checks ``expression`` before the reached unit executes. An unavailable
    result adds no edits. Reuse is detected from actual pending helper edits,
    so restoring a scope checkpoint cannot leave a stale helper cache behind.
    """
    if builder.config.policy != "auto":
        return ComputeIdentityGuard()
    analysis = builder.analysis
    limit = analysis.operation_limit
    selected = []
    for procedure in procedures:
        if procedure not in selected:
            selected.append(procedure)
        if len(selected) > limit:
            return _unavailable("compute identity exceeds bounded reached source selection")
    requested = []
    for procedure in selected:
        regional = builder.inline_for(procedure)
        prepared = regional.generated if regional is not None else builder.generated
        if procedure not in prepared:
            return _unavailable("compute identity requires prepared numerical artifacts: " + procedure)
        generated = prepared[procedure]
        scoped = generated.scoped if generated is not None else None
        if scoped is None:
            continue
        for unit in scoped.get("planning", {}).get("units", ()):
            model = unit.get("compute_model")
            if model is not None:
                requested.append((procedure, regional, model))
                if len(requested) > limit:
                    return _unavailable("compute identity exceeds bounded reached planning units")
    if not requested:
        return ComputeIdentityGuard()
    try:
        caller = builder.entry if caller is None else analysis.routines.get(caller) if isinstance(caller, str) else caller
        groups, object_identities, checked = {}, {}, set()
        if caller is None or analysis.routines.get(getattr(caller, "qualified", None)) is not caller:
            raise CompilationError("compute identity caller lacks original procedure authority")
        analysis._require_original(caller.qualified)
        checked.add(caller.qualified)
        caller_scope = caller.scope
        for procedure, regional, model in requested:
            routine = regional.builder.entry if regional is not None else analysis.routines.get(procedure)
            if routine is None or analysis.routines.get(routine.qualified) is not routine:
                raise CompilationError("compute model lacks original procedure authority: " + procedure)
            if routine.qualified not in checked:
                analysis._require_original(routine.qualified)
                checked.add(routine.qualified)
            module = _module(analysis, routine)
            source = str(module.path)
            source_digest = analysis.sources.get(source)
            if source_digest is None or routine.scope.path != module.path:
                raise CompilationError("compute model lacks original source-object identity")
            identity = model.get("fortran")
            if not isinstance(identity, dict):
                raise CompilationError("compute model lacks calibrated native Fortran identity")
            version, semantic = identity.get("compiler_version"), identity.get("semantic_options")
            _literal_parts(version)
            _literal_parts(semantic, semantic=True)
            runtime_schedule = model.get("runtime_schedule")
            if (runtime_schedule is not None and (not isinstance(runtime_schedule, dict)
                    or runtime_schedule != {"kind": "static", "chunk": 0}
                    or type(runtime_schedule.get("chunk")) is not int)):
                raise CompilationError("unsupported original runtime schedule requirement")
            key = (source, module.module, runtime_schedule is not None)
            if key[:2] in object_identities and object_identities[key[:2]] != (version, semantic):
                raise CompilationError("conflicting calibrated identities for one original source object")
            object_identities[key[:2]] = version, semantic
            groups.setdefault(key, (module, (version, semantic), set(), source_digest))[2].add(procedure)
        if len(groups) > analysis.procedure_limit:
            raise CompilationError("compute identity exceeds bounded original source objects")
        if "fort_scoped_team_observer" in analysis.modules:
            raise CompilationError("original source shadows the compute compatibility runtime module")
        requirements, edits, imports, expressions = [], [], [], []
        for (source, module_name, runtime_schedule), (module, (version, semantic), names, source_digest) in sorted(groups.items()):
            schedule_identity = "\0runtime-static-chunk0-v1" if runtime_schedule else ""
            name = "fort_compute_identity_" + sha256((source + "\0" + module_name + schedule_identity).encode()).hexdigest()[:12]
            if _conflicts(analysis, module, name):
                raise CompilationError("compute identity helper conflicts with original module names")
            if _conflicts(analysis, caller_scope, name):
                raise CompilationError("compute identity helper conflicts with original caller names")
            code = _helper(name, version, semantic, runtime_schedule=runtime_schedule)
            previous = [text for _, _, text in builder.edits.get(module.path, ())
                        if text.startswith("logical function " + name + "(")]
            if previous and previous != [code]:
                raise CompilationError("compute identity helper changed within one source object")
            if not previous:
                edits.append((module, name, code))
            if caller_scope.module != module_name:
                imports.append("use " + module_name + ", only: " + name)
            expressions.append(name + "()")
            requirements.append(ComputeIdentityRequirement(source, source_digest, module_name,
                tuple(sorted(names)), name, version, semantic,
                {"kind": "static", "chunk": 0} if runtime_schedule else None))
    except (CompilationError, AttributeError, TypeError) as error:
        return _unavailable(str(error))
    # Every identity and source is checked before the first source mutation.
    for module, name, code in edits:
        builder.append_procedure(module, name, code)
    return ComputeIdentityGuard(required=True, imports=tuple(imports), expression=" .and. &\n    ".join(expressions),
                                requirements=tuple(requirements))
