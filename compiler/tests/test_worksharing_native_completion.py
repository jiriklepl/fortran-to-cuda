"""Native subunit coherence needs its own authenticated original-team proof."""

from dataclasses import replace

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk
import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.regions import extract_region
from compiler.tests.test_native_completion_tokens import PROCEDURE, fixture


UNIT = '!$omp do\ndo i=1,n\na(-2,i)=b(1,i)\nenddo\n!$omp end do\n'


def example(tmp_path, body=UNIT, **kwargs):
    path, analysis, nodes = fixture(tmp_path, body, **kwargs)
    loops = tuple(walk(nodes, F.Block_Nonlabel_Do_Construct))
    return path, analysis, nodes, loops


def test_registered_native_subunit_refines_exact_faces_and_retains_team(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, UNIT + UNIT.replace('-2,i', '2,i'))
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    assert proof.validate(analysis, PROCEDURE, loops) is proof
    assert proof.original_nodes == loops
    assert tuple(unit[1] for unit in proof.units) == loops
    assert proof.parent is analysis.joined_completion(PROCEDURE, nodes)
    record = proof.public()
    assert record['native_effects_authority']
    assert not record['standalone_execution_authority']
    assert not record['gpu_legality_established']
    assert not record['requires_serial_caller']
    assert record['worksharing_units'] == 2
    assert 'team barrier' in record['coherence_protocol']
    sections = analysis.native_sections_for_nodes(PROCEDURE, loops, completion=proof)
    assert sections.available, sections.reason
    output = next(resource for resource in sections.resources if resource.resource == 'argument::a')
    assert len(output.writes) == len(output.overwrites) == 2
    assert [box.axes[0].lower.value for box in output.writes] == [-2, 2]
    assert all(box.axes[0].point for box in output.writes)
    record['private_resources'].clear()
    assert proof.private_roots


def test_whole_group_authority_still_cannot_authorize_one_native_subunit(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, UNIT * 2)
    parent = analysis.joined_completion(PROCEDURE, nodes)
    with pytest.raises(CompilationError, match='whole original joined-group authority'):
        analysis.native_sections_for_nodes(PROCEDURE, (loops[0],), completion=parent)
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[0],))
    assert analysis.native_sections_for_nodes(PROCEDURE, (loops[0],), completion=proof).available


def test_numerical_and_native_tokens_cannot_authorize_each_others_role(tmp_path):
    _, analysis, nodes, loops = example(tmp_path)
    numerical = analysis.worksharing_completion(PROCEDURE, nodes, loops)
    native = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    with pytest.raises(CompilationError, match='proof token'):
        analysis.native_sections_for_nodes(PROCEDURE, loops, completion=numerical)
    with pytest.raises(CompilationError, match='participation proof'):
        extract_region(analysis, analysis.routines[PROCEDURE], loops[0], worksharing=native)


@pytest.mark.parametrize('copy_token', [lambda proof: replace(proof), lambda proof: proof.public()])
def test_copied_native_token_cannot_grant_effect_authority(tmp_path, copy_token):
    _, analysis, nodes, loops = example(tmp_path)
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    with pytest.raises(CompilationError, match='(proof token|original-team authority)'):
        analysis.native_sections_for_nodes(PROCEDURE, loops, completion=copy_token(proof))


def test_foreign_or_changed_original_source_invalidates_token(tmp_path):
    path, analysis, nodes, loops = example(tmp_path)
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    independent = SourceEffects([path])
    foreign_loops = tuple(walk(independent.routines[PROCEDURE].execution, F.Block_Nonlabel_Do_Construct))
    with pytest.raises(CompilationError, match='original-team authority'):
        independent.native_sections_for_nodes(PROCEDURE, foreign_loops, completion=proof)
    path.write_text(path.read_text().replace('end do', 'end do nowait'))
    with pytest.raises(CompilationError, match='changed'):
        analysis.native_sections_for_nodes(PROCEDURE, loops, completion=proof)


def test_token_does_not_authorize_a_sibling_or_skipped_unit(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, UNIT * 3)
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[0],))
    with pytest.raises(CompilationError, match='original-team authority'):
        analysis.native_sections_for_nodes(PROCEDURE, (loops[1],), completion=proof)
    with pytest.raises(CompilationError, match='contiguous ordered set'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[0], loops[2]))
    with pytest.raises(CompilationError, match='original source order'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, tuple(reversed(loops)))


@pytest.mark.parametrize('body,reason', [
    (UNIT.replace('end do', 'end do nowait'), 'NOWAIT'),
    ('if(i>0) then\n' + UNIT + 'endif', 'condition is private'),
    (UNIT.replace('a(-2,i)=b(1,i)', 'call opaque(a)'), 'source-call proof'),
])
def test_unfinished_or_nonuniform_original_teams_remain_boundaries(tmp_path, body, reason):
    _, analysis, nodes, loops = example(tmp_path, body)
    with pytest.raises(CompilationError, match=reason):
        analysis.worksharing_native_completion(PROCEDURE, nodes, loops)


def test_optional_end_do_uses_original_implicit_completion(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, UNIT.replace('!$omp end do\n', ''))
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    assert proof.units[0][2] is None
    assert analysis.native_sections_for_nodes(PROCEDURE, loops, completion=proof).available


def test_complete_uniform_branch_retains_possible_writes_without_overwrite(tmp_path):
    body = 'if(flag) then\n' + UNIT + 'else\n' + UNIT.replace('-2,i', '2,i') + 'endif'
    _, analysis, nodes, loops = example(tmp_path, body)
    branch, = walk(nodes, F.If_Construct)
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, (branch,))
    sections = analysis.native_sections_for_nodes(PROCEDURE, (branch,), completion=proof)
    assert sections.available, sections.reason
    output = next(resource for resource in sections.resources if resource.resource == 'argument::a')
    assert len(output.writes) == 2
    assert not output.overwrites
    with pytest.raises(CompilationError, match='reached guarded segment'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, loops)


def test_original_private_array_never_becomes_a_shared_native_capture(tmp_path):
    body = UNIT.replace('a(-2,i)=b(1,i)', 'local(0)=b(1,i)\na(-2,i)=local(0)')
    _, analysis, nodes, loops = example(tmp_path, body, declaration='real(8)::local(-1:1)',
                                       opening='parallel private(i,local)')
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    sections = analysis.native_sections_for_nodes(PROCEDURE, loops, completion=proof, capture_locals=True)
    assert sections.available, sections.reason
    assert {resource.resource for resource in sections.resources} == {'argument::a', 'argument::b'}
    assert 'grouped::step::local' in proof.private_roots


def test_indirect_native_work_retains_conservative_effects_without_gpu_authority(tmp_path):
    body = UNIT.replace('a(-2,i)=b(1,i)', 'j=int(b(1,i))\na(j,i)=b(2,i)')
    _, analysis, nodes, loops = example(tmp_path, body, opening='parallel private(i,j)')
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, loops)
    sections = analysis.native_sections_for_nodes(PROCEDURE, loops, completion=proof)
    assert not sections.available
    summary = analysis.segment_summary(PROCEDURE, loops, capture_locals=True)
    assert summary['complete'], summary['reasons']
    writes = [operation for operation in summary['operations']
              if operation.get('resource') == 'argument::a' and operation['kind'] == 'write']
    assert writes and all(operation['section'] == 'whole' for operation in writes)
    assert not proof.public()['gpu_legality_established']


def test_serial_inner_do_cannot_claim_native_worksharing_completion(tmp_path):
    body = UNIT.replace('a(-2,i)=b(1,i)', 'do j=1,2\na(-2,i)=b(j,i)\nenddo')
    _, analysis, nodes, loops = example(tmp_path, body)
    with pytest.raises(CompilationError, match='complete associated original DOs'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[1],))


def test_empty_selection_does_not_issue_a_completion_token(tmp_path):
    _, analysis, nodes, _ = example(tmp_path)
    with pytest.raises(CompilationError, match='bounded nonempty'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, ())


def test_nowait_in_an_unselected_sibling_prevents_coherence_cuts(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, UNIT.replace('end do', 'end do nowait') + UNIT)
    with pytest.raises(CompilationError, match='NOWAIT'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[-1],))


def test_ancestor_outside_joined_team_cannot_borrow_native_authority(tmp_path):
    path, _, _, _ = example(tmp_path)
    path.write_text(path.read_text().replace('!$omp parallel', 'if(flag) then\n!$omp parallel', 1)
                    .replace('!$omp end parallel', '!$omp end parallel\nendif', 1))
    analysis = SourceEffects([path])
    branch, = walk(analysis.routines[PROCEDURE].execution, F.If_Construct)
    joined = tuple(branch.content[1:-1])
    with pytest.raises(CompilationError, match='inside its original joined team'):
        analysis.worksharing_native_completion(PROCEDURE, joined, (branch,))


def test_combined_parallel_do_retains_whole_group_authority_only(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, 'do i=1,n\na(-2,i)=b(1,i)\nenddo',
                                       opening='parallel do private(i)', ending='end parallel do')
    assert analysis.joined_completion(PROCEDURE, nodes)
    with pytest.raises(CompilationError, match='separate original PARALLEL and DO'):
        analysis.worksharing_native_completion(PROCEDURE, nodes, loops)


def test_private_coordinate_carried_from_an_earlier_do_is_not_a_coordinator_bound(tmp_path):
    body = UNIT.replace('a(-2,i)=b(1,i)', 'j=i\na(-2,i)=b(1,i)')
    body += UNIT.replace('a(-2,i)=b(1,i)', 'a(-2,i+j)=b(1,i)')
    _, analysis, nodes, loops = example(tmp_path, body, opening='parallel private(i,j)')
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[1],))
    sections = analysis.native_sections_for_nodes(PROCEDURE, (loops[1],), completion=proof)
    assert not sections.available
    assert 'thread-private state: grouped::step::j' in sections.reason
    summary = analysis.segment_summary(PROCEDURE, (loops[1],), capture_locals=True)
    assert summary['complete'], summary['reasons']


def test_private_numerical_value_does_not_prevent_uniform_native_footprints(tmp_path):
    body = UNIT.replace('a(-2,i)=b(1,i)', 'j=i\na(-2,i)=b(1,i)')
    body += UNIT.replace('a(-2,i)=b(1,i)', 'a(-2,i)=b(1,i)+real(j,8)')
    _, analysis, nodes, loops = example(tmp_path, body, opening='parallel private(i,j)')
    proof = analysis.worksharing_native_completion(PROCEDURE, nodes, (loops[1],))
    sections = analysis.native_sections_for_nodes(PROCEDURE, (loops[1],), completion=proof)
    assert sections.available, sections.reason
    assert all(not dependency.resource.endswith('::i') and not dependency.resource.endswith('::j')
               for resource in sections.resources for dependency in resource.dependencies)
