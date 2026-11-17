"""Exact entry reads for unchanged native scatter work after a GPU prefix."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.indirect_sections import analyze_indirect_sections
from compiler.frontend.reached_read_requirements import native_read_sections, prove_native_read_requirements
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner
from compiler.scopes.source_reads import validate_native_source_reads
from compiler.tests.test_indirect_native_sections import fixture
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_original_interval_owner import builder, call_pair
from compiler.tests.test_selection_activation import PREFIX

CORRECTION = "s(sites(q)%index)=s(sites(q)%index)+a(sites(q)%index)"
TEAM = """integer::i,q
visits=visits+1
!$omp parallel private(i,q)
!$omp do schedule(runtime)
do i=0,n-1
s(i)=v(i)+a(i)
enddo
!$omp end do
if(correcting) then
!$omp do
do q=1,size(sites)
""" + CORRECTION + """
enddo
!$omp end do
endif
!$omp do schedule(runtime)
do i=0,n-1
o(i)=o(i)+s(i)
enddo
!$omp end do
!$omp end parallel
"""


def emit_native_team_fixture(path, *, read_modify_write=True):
    source = SOURCE.replace("integer::visits=0", """integer::visits=0
type site_t
integer::index
end type
type(site_t),target::sites(2)""")
    start = source.index("integer::i", source.index("subroutine momentum"))
    stop = source.index("end subroutine", start)
    source = source[:start] + TEAM + source[stop:]
    if not read_modify_write:
        source = source.replace(CORRECTION, "s(sites(q)%index)=a(sites(q)%index)")
    selected = builder(path, source=source, prefix=PREFIX + "sites(1)%index=0\nsites(2)%index=-1\n")
    selected.facts["captures"]["interval_fields::out"]["initialized"] = "none"
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    return selected, owner, outputs, report


def native_correction(owner):
    return next((member, operation) for member in owner.execution_plan.members for unit in member.units
                for operation in unit.scope.native
                if getattr(operation, "indirect_sections", None) is not None
                and operation.indirect_sections.available)


@pytest.mark.parametrize("body", [
    "a(markers(q)%row,markers(q)%column)=1.d0",
    "a(markers(q)%row,markers(q)%column)=a(markers(q)%row+1,markers(q)%column)",
    "a(markers(q)%row,markers(q)%column)=merge(a(markers(q)%row,markers(q)%column),0.d0,n>0)",
    "a(markers(q)%row,markers(q)%column)=abs(a(markers(q)%row,markers(q)%column))",
    "a(markers(q)%row,markers(q)%column)=1.d0\n"
    "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+1.d0",
])
def test_communication_union_alone_is_not_an_entry_read_precondition(tmp_path, body):
    analysis, communication = fixture(tmp_path, "do q=1,size(markers)\n" + body + "\nenddo")
    assert communication.available, communication.reason
    routine = analysis.routines["geometry::adjust"]
    reads = analyze_indirect_sections(analysis, routine.qualified, routine.execution.content,
                                      require_entry_reads=True)
    assert not reads.available
    assert "entry reads" in reads.reason


def test_repeated_updates_keep_the_same_original_entry_requirement(tmp_path):
    update = "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value"
    analysis, communication = fixture(tmp_path, "do q=1,size(markers)\n" + update + "\n" + update + "\nenddo")
    routine = analysis.routines["geometry::adjust"]
    reads = analyze_indirect_sections(analysis, routine.qualified, routine.execution.content,
                                      require_entry_reads=True)
    assert reads.available, reads.reason
    assert reads.public() == communication.public()


def test_wrong_numerical_completion_cannot_authorize_native_entry_reads(tmp_path):
    source = """!$omp parallel private(q)
!$omp do
do q=1,size(markers)
a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value
enddo
!$omp end do
!$omp end parallel"""
    analysis, _ = fixture(tmp_path, source)
    routine = analysis.routines["geometry::adjust"]
    loops = tuple(walk(routine.execution, F.Block_Nonlabel_Do_Construct))
    proof = analysis.worksharing_completion(routine.qualified, routine.execution.content, loops)
    with pytest.raises(CompilationError, match="unchanged worksharing completion"):
        prove_native_read_requirements(analysis, routine.qualified, loops, completion=proof)
