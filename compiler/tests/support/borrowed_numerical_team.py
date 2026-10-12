"""Fresh fixtures for reusable numerical helpers with an original team.

The source-only writers are shared with the CUDA tests. Every builder call
creates fresh source analysis and proof objects.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

ENTRY = "amber_owner::advance"
INITIALIZER = "amber_planes::initialize_planes"
SCALER = "amber_planes::scale_planes"
FACT = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
DUMMY_LOWER_BOUNDS = {"a": (-2, -1, 0), "work": (-2, -1, 0), "b": (-4, -3, -2),
                      "c": (-7, -6, -5), "d": (2, -9, 4)}

HELPER_SOURCE = """module amber_planes
implicit none
contains
subroutine initialize_planes(values,seed)
real(8),contiguous,intent(out)::values(:,:,:)
integer,intent(in)::seed
integer::plane
!$omp parallel do private(plane)
do plane=1,size(values,3)
values(:,:,plane)=seed
enddo
!$omp end parallel do
end subroutine
subroutine scale_planes(values,factor)
real(8),contiguous,intent(inout)::values(:,:,:)
real(8),intent(in)::factor
integer::plane
!$omp parallel do private(plane)
do plane=1,size(values,3)
values(:,:,plane)=values(:,:,plane)*factor
enddo
!$omp end parallel do
end subroutine
end module
"""


def _loop(statement):
    return "do k=1,p\ndo j=1,m\ndo i=1,n\n" + statement + "\nenddo\nenddo\nenddo\n"


def _element(name):
    offsets = tuple(low - origin for low, origin in zip(DUMMY_LOWER_BOUNDS[name], (-2, -1, 0), strict=True))
    indices = [variable + (f"+{offset}" if offset > 0 else str(offset) if offset else "")
               for variable, offset in zip(("i", "j", "k"), offsets, strict=True)]
    return name + "(" + ",".join(indices) + ")"


def _owner_source():
    source = """module amber_owner
use amber_planes,only:reset=>initialize_planes,scale=>scale_planes
use,intrinsic::ieee_exceptions,only:ieee_get_halting_mode,ieee_invalid
implicit none
integer::owner_visits=0
contains
subroutine advance(a,work,b,c,d,n,m,p,gain)
real(8),contiguous,intent(in)::a(-2:,-1:,0:)
real(8),contiguous,intent(inout)::work(-2:,-1:,0:)
real(8),contiguous,intent(out)::b(-4:,-3:,-2:),c(-7:,-6:,-5:),d(2:,-9:,4:)
real(8),intent(in)::gain
real(8)::scale_factor
integer,intent(in)::n,m,p
integer::i,j,k
logical::original_mask
owner_visits=owner_visits+1
call ieee_get_halting_mode(ieee_invalid,original_mask)
"""
    source += _loop("work(i,j,k)=work(i,j,k)+a(i,j,k)+1.0_8")
    for target, coefficient in zip(("b", "c", "d"), ("0.25_8", "0.5_8", "0.75_8"), strict=True):
        element = _element(target)
        source += f"call reset({target},0)\n"
        source += _loop(element + "=" + element + "+a(i,j,k)")
        source += _loop(element + "=" + element + "+work(i,j,k)")
        source += f"scale_factor=gain+{coefficient}\ncall scale({target},scale_factor)\n"
    source += _loop("work(i,j,k)=work(i,j,k)+" + "+".join(_element(name) for name in ("b", "c", "d")))
    return source + "end subroutine\nend module\n"


OWNER_SOURCE = _owner_source()


def write_borrowed_team_sources(directory, *, precision=8, helper_source=HELPER_SOURCE, owner_source=OWNER_SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("amber_planes.f90", "amber_owner.f90"))
    for path, source in zip(paths, (helper_source, owner_source), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})").replace("_8", f"_{precision}"))
    return paths


def borrowed_team_facts(paths, *, captures=None):
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::" + name: {**deepcopy(FACT), "initialized": initialized}
                          for name, initialized in (("a", "whole"), ("work", "whole"),
                                                    ("b", "none"), ("c", "none"), ("d", "none"))}}
    for root, values in (captures or {}).items():
        facts["captures"][root].update(values)
    return facts


def borrowed_team_fixture(directory, *, captures=None, **source_options):
    paths = write_borrowed_team_sources(directory, **source_options)
    builder = ScopeBuilder(paths, ENTRY, facts=borrowed_team_facts(paths, captures=captures),
                           options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    return paths, builder


def emit_borrowed_team_fixture(directory, **options):
    paths, builder = borrowed_team_fixture(directory, **options)
    outputs, report = builder.run()
    return paths, outputs, report, builder


def borrowed_team_launch_counts(builder, report):
    """Bind launch expectations to complete original coverage and public units."""
    from fparser.two.utils import walk

    from compiler.scopes.segments import statement_span

    originals = tuple(node for node in builder.entry.execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(originals) == 8  # Prefix, two producers per output, final consumer.
    registered = tuple(region for procedure, region in builder.inline.regions.items()
                       if procedure in builder.inline.used)
    covered = {id(node) for region in registered for original in region.nodes for node in walk(original)}
    assert all(id(loop) in covered for loop in originals)
    public = report["inline_numerical_regions"]["regions"]
    spans = tuple(statement_span(loop) for loop in originals)
    assert all(any(region["used"] and region["first_line"] <= span[0]
                   and span[1] <= region["last_line"] for region in public) for span in spans)
    owner, = report["scopes"]
    root_leaves = {leaf for leaf in owner["gpu_leaves"] if leaf.startswith(ENTRY + "#")}
    root_launches = sum(len(builder.generated[leaf].scoped["planning"]["units"]) for leaf in root_leaves)
    children = {child["procedure"]: child for child in report["borrowed_source_coordinators"]}
    units = {}
    for procedure in (INITIALIZER, SCALER):
        leaf, = children[procedure]["gpu_leaves"]
        units[procedure] = len(builder.generated[leaf].scoped["planning"]["units"])
        assert units[procedure] == 1
    calls = [call for call in report["resolved_calls"]
             if call["caller"] == ENTRY and call["procedure"] in units]
    assert len(calls) == 6
    return root_launches, sum(units[call["procedure"]] for call in calls)


@dataclass(frozen=True)
class EmittedBorrowedTeam:
    original_sources: tuple[tuple[str, str], ...]
    artifacts: tuple[tuple[str, str], ...]
    manifest_json: str

    @classmethod
    def capture(cls, paths, outputs, report):
        return cls(tuple((str(path), path.read_text()) for path in paths), tuple(sorted(outputs.items())),
                   json.dumps(report, sort_keys=True, separators=(",", ":")))

    def report(self):
        return json.loads(self.manifest_json)

    def source(self, index):
        report = self.report()
        return dict(self.artifacts)[report["sources"][self.original_sources[index][0]]["replacement"]]
