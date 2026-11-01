"""SECTION inventory is subordinate to a complete original parent join."""
import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.atomic_worksharing_team import atomic_worksharing_alternatives, prove_atomic_worksharing_team
from compiler.frontend.native_completion import _joined_completion_facts
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


ENTRY = 'section_fixture::step'
SOURCE = '''module section_fixture
contains
subroutine step(a,n,flag)
real(8),intent(inout)::a(-2:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
!$omp parallel private(i)
if(flag)then
!$omp sections
!$omp section
do i=n+1,n+3
a(i)=a(i-n)
enddo
!$omp section
do i=-2,0
a(i)=a(i+n)
enddo
!$omp end sections nowait
endif
!$omp end parallel
end subroutine
end module
'''


def issue(tmp_path, text=SOURCE):
    path = tmp_path / 'sections.f90'
    path.write_text(text)
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    nodes, = [candidate.original_nodes for candidate in grouped_original_sequence(routine.execution.content, {})
              if isinstance(candidate, NativeGroupCandidate)]
    return analysis, routine, nodes, prove_atomic_worksharing_team(analysis, routine, nodes)


def test_exact_section_inventory_grants_no_independent_worksharing_completion(tmp_path):
    analysis, routine, nodes, team = issue(tmp_path)
    record = team.public()
    assert record['version'] == 3 and record['conditional_affine_context_required'] is True
    assert record['alternatives'] == [[0, 1], []]
    assert record['parent_completion']['native_only'] is True
    assert record['parent_completion']['section_independence_established'] is False
    assert record['parent_completion']['internal_cuts_authorized'] is False
    assert record['parent_completion']['original_section_count'] == 2
    assert all(unit['unit_kind'] == 'section' and unit['nowait'] for unit in record['units'])
    assert len({unit['section_group_node_id'] for unit in record['units']}) == 1
    assert len({unit['section_directive_node_id'] for unit in record['units']}) == 2
    assert team.validate() is team
    with pytest.raises(CompilationError, match='worksharing cuts'):
        _joined_completion_facts(analysis, ENTRY, nodes, worksharing=[])
    alternative = atomic_worksharing_alternatives(team)[0]
    assert alternative.public()['individual_unit_completion_authorized'] is False
    provider = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', host_threads=4), environment_root=ENTRY)
    with pytest.raises(CompilationError, match='guarded conditional context'):
        provider.prepare_atomic_alternative(alternative)


@pytest.mark.parametrize('change', [
    ('!$omp section\ndo i=n+1', 'do i=n+1'),
    ('!$omp end sections nowait', '!$omp end sections nowait schedule(static)'),
    ('!$omp parallel private(i)', '!$omp parallel'),
    ('!$omp section\ndo i=n+1', '!$omp section\n!$omp do\ndo i=n+1'),
    ('a(i)=a(i-n)', 'call unknown(a)'),
])
def test_unknown_sections_and_private_storage_do_not_broaden(tmp_path, change):
    with pytest.raises(CompilationError):
        issue(tmp_path, SOURCE.replace(*change))


@pytest.mark.parametrize('change', ['foreign_unit_marker', 'foreign_group_marker', 'kind', 'registry'])
def test_section_identity_stays_original_and_registered(tmp_path, change):
    analysis, _routine, _nodes, team = issue(tmp_path)
    if change == 'registry':
        analysis._atomic_worksharing_teams.pop(team.identity)
    elif change == 'foreign_unit_marker':
        object.__setattr__(team.units[0], 'opening', team.units[1].opening)
    elif change == 'foreign_group_marker':
        object.__setattr__(team.units[0], 'group_opening', team.units[0].opening)
    else:
        object.__setattr__(team.units[0], 'unit_kind', 'worksharing_do')
    with pytest.raises(CompilationError):
        team.validate()
