"""Fresh source-backed numerical fixtures and immutable emitted artifacts.

Every fixture call creates new analysis, builder and proof objects. Only source
strings and serialized emission snapshots may be shared by read-only tests.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_windows import plan_reached_window
from compiler.scopes.combined_team import CombinedNumericalTeam
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder

FACT = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}

ENTRY = "quartz_owner::advance"
CHILD = "quartz_numerics::bridge"
ROOTS = ("argument::a", "argument::b", "argument::c")

MATH_SOURCE = """module quartz_math
implicit none
contains
pure subroutine profile_cell(g,s,value,index)
real(8),intent(out)::g(2),s
real(8),intent(in)::value
integer,intent(in)::index
g(1)=value
g(2)=real(index+4,8)
s=dot_product(g,g)
end subroutine
end module
"""

CHILD_SOURCE = """module quartz_numerics
use quartz_math,only:cell_formula=>profile_cell
implicit none
integer::bridge_visits=0,middle_visits=0
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i,block
real(8)::g(2),s
bridge_visits=bridge_visits+1
i=0
if(tile==-5) then
!$omp parallel do private(i) schedule(static)
do i=-3,n
b(i)=b(i)+a(i)+real(i,8)
enddo
!$omp end parallel do
endif
!$omp parallel do private(block,i,g,s) schedule(static)
do block=-3,n,tile
do i=block,min(block+tile-1,n)
call cell_formula(g,s,a(i),i)
b(i)=s+real(i,8)
enddo
enddo
!$omp end parallel do
if(n>=-3) then
call touch_face(b)
endif
do i=-3,n
c(i)=2.0_8*b(i)+real(i+1,8)
enddo
end subroutine
subroutine touch_face(b)
real(8),intent(inout)::b(-4:)
middle_visits=middle_visits+1
b(-3)=b(-3)+0.5_8
end subroutine
end module
"""

OWNER_SOURCE = """module quartz_owner
use quartz_numerics,only:renamed_bridge=>bridge
implicit none
integer::prefix_visits=0,tail_visits=0
contains
subroutine advance(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
prefix_visits=prefix_visits+1
call renamed_bridge(a,b,c,n,tile)
tail_visits=tail_visits+1
end subroutine
end module
"""


def reached_numerical_fixture(directory, *, precision=8, device_budget=None, provider_enabled=True,
                              math_source=MATH_SOURCE, child_source=CHILD_SOURCE, owner_source=OWNER_SOURCE,
                              captures=None):
    """The caller defines a/b; only c depends on source-established coverage."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("quartz_math.f90", "quartz_numerics.f90", "quartz_owner.f90"))
    for path, source in zip(paths, (math_source, child_source, owner_source), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {ROOTS[0]: deepcopy(FACT), ROOTS[1]: deepcopy(FACT),
                          ROOTS[2]: {**deepcopy(FACT), "initialized": "none", "lifetime": "selected_window"}}}
    for root, values in (captures or {}).items():
        facts["captures"][root].update(values)
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    originals = tuple(node for node in builder.entry.execution.content if type(node).__name__ == "Call_Stmt")
    assert len(originals) == 1
    provider = NumericalPreparationProvider(builder.analysis, options=builder.options, config=builder.config,
        environment_root=builder._numerical_environment_root, immutable_inputs={}) if provider_enabled else None
    options = {"numerical_provider": provider} if provider_enabled else {}
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, facts, **options)
    return paths, builder, proof, originals, provider


def emit_reached_numerical_fixture(directory, **options):
    paths, builder, proof, originals, _provider = reached_numerical_fixture(directory, **options)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return paths, outputs, report, owner


ENVIRONMENT_USE = "use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid\n"
ENVIRONMENT_TEAM = """!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
"""


def emit_combined_team_fixture(directory, **options):
    """Keep the native environment setter and force original-team admission."""
    source = options.pop("child_source", CHILD_SOURCE)
    assert "implicit none\n" in source
    assert "i=0\n" in source
    source = environment_source(source)
    return emit_reached_numerical_fixture(directory, child_source=source, **options)


def combined_units(owner):
    return tuple(unit for unit in owner.members[CHILD].units if isinstance(unit.scope, CombinedNumericalTeam))


def emitted_child(paths, outputs, report):
    return outputs[report["sources"][str(paths[1])]["replacement"]]


def team_text(text):
    """The original setter has no generated environment-vote marker."""
    body = text.index("call cell_formula(g,s,a(i),i)")
    marker = text.rfind("fort_team_environment_supported = .true.", 0, body)
    assert marker >= 0
    first = text.rfind("!$omp parallel", 0, marker)
    last = text.index("!$omp end parallel", marker) + len("!$omp end parallel")
    assert first >= 0
    return text[first:last]


def environment_source(source=CHILD_SOURCE):
    """Retain the original native setter before the numerical groups."""
    return source.replace("implicit none\n", ENVIRONMENT_USE + "implicit none\n", 1).replace(
        "i=0\n", "i=0\n" + ENVIRONMENT_TEAM, 1)


@dataclass(frozen=True)
class EmittedNumericalText:
    """Read-only emission data, without analysis, builder or proof authority."""

    original_sources: tuple[tuple[str, str], ...]
    artifacts: tuple[tuple[str, str], ...]
    manifest_json: str
    observations_json: str

    @classmethod
    def capture(cls, paths, outputs, report, *, observations=None):
        return cls(tuple((str(path), path.read_text()) for path in paths),
                   tuple(sorted(outputs.items())),
                   json.dumps(report, sort_keys=True, separators=(",", ":")),
                   json.dumps(observations or {}, sort_keys=True, separators=(",", ":")))

    def manifest(self):
        """Each caller owns its mutable public JSON copy."""
        return json.loads(self.manifest_json)

    def observations(self):
        return json.loads(self.observations_json)

    def child_text(self):
        path, _source = self.original_sources[1]
        replacement = self.manifest()["sources"][path]["replacement"]
        return dict(self.artifacts)[replacement]
