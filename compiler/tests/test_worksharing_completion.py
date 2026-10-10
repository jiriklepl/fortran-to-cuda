"""A numerical child loop cannot borrow whole-team authority implicitly."""

from dataclasses import replace

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk
import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.regions import extract_region
from compiler.tests.test_native_completion_tokens import fixture, PROCEDURE


BODY = '''!$omp do
do i=1,n
a(-2,i)=2*b(1,i)
enddo
!$omp end do
if(flag) then
!$omp do
do i=1,n
a(-2,i)=a(-2,i)+b(2,i)
enddo
!$omp end do
endif'''


def example(tmp_path, body=BODY, **kwargs):
    path, analysis, nodes = fixture(tmp_path, body, **kwargs)
    loops = tuple(walk(nodes, F.Block_Nonlabel_Do_Construct))
    return path, analysis, nodes, loops


def test_uniform_reached_unit_needs_separate_participation_proof(tmp_path):
    _, analysis, nodes, loops = example(tmp_path)
    routine = analysis.routines[PROCEDURE]
    for loop in loops:
        with pytest.raises(CompilationError, match='complete original source region'):
            extract_region(analysis, routine, loop)
        proof = analysis.worksharing_completion(PROCEDURE, nodes, (loop,))
        assert proof.validate(analysis, PROCEDURE, (loop,)) is proof
        extracted = extract_region(analysis, routine, loop, worksharing=proof)
        assert extracted.completion['caller_contract'].startswith('all members of the original team')
        assert extracted.completion['standalone_execution_authority'] is False
        assert extracted.completion['native_effects_authority'] is False
        assert 'omp' not in extracted.source.lower()
        assert extracted.private_scalars == ('i',)


def test_whole_team_extraction_cannot_drop_a_uniform_sibling_branch(tmp_path):
    _, analysis, nodes, _ = example(tmp_path)
    with pytest.raises(CompilationError, match='every original operation'):
        extract_region(analysis, analysis.routines[PROCEDURE], nodes)


def test_optional_end_do_retains_implicit_worksharing_completion(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, BODY.replace('!$omp end do\n', ''))
    proof = analysis.worksharing_completion(PROCEDURE, nodes, (loops[0],))
    assert proof.public()['join'].startswith('implicit')


def test_proven_subregion_lowers_through_existing_collective_backend(tmp_path):
    from compiler.driver.pipeline import prepare_function
    from compiler.emission import generate_sources
    from compiler.frontend import lower_source
    from compiler.offload.config import OffloadConfig

    _, analysis, nodes, loops = example(tmp_path)
    proof = analysis.worksharing_completion(PROCEDURE, nodes, (loops[0],))
    region = extract_region(analysis, analysis.routines[PROCEDURE], loops[0], worksharing=proof)
    function, plan = prepare_function(lower_source(region.source, region.entry, source_name='worksharing.f90'))
    assert plan.regions
    generated = generate_sources(function, plan, memory_model='scoped',
        offload_config=OffloadConfig(policy='sections', collective=True, host_threads=4))
    team = generated.scoped['team']
    assert team['available'] and team['fortran_procedure'] == 'run_team'
    assert team['host_threads'] == 4 and team['coordinator'] == 'master'
    assert not team['automatic_estimate_available']


def test_inner_serial_loop_cannot_impersonate_its_associated_worksharing_do(tmp_path):
    body = '!$omp do\ndo i=1,n\ndo j=1,2\na(-2,i)=b(j,i)\nenddo\nenddo\n!$omp end do'
    _, analysis, nodes, loops = example(tmp_path, body)
    with pytest.raises(CompilationError, match='exactly one associated original DO'):
        analysis.worksharing_completion(PROCEDURE, nodes, (loops[1],))


@pytest.mark.parametrize('change,reason', [
    (lambda body: body.replace('end do', 'end do nowait', 1), 'NOWAIT'),
    (lambda body: body.replace('if(flag)', 'if(i>0)'), 'condition is private'),
    (lambda body: body.replace('a(-2,i)=2*b(1,i)', 'call opaque(a)'), 'source-call proof'),
])
def test_unfinished_work_or_nonuniform_entry_does_not_authorize_cuts(tmp_path, change, reason):
    _, analysis, nodes, loops = example(tmp_path, change(BODY))
    with pytest.raises(CompilationError, match=reason):
        analysis.worksharing_completion(PROCEDURE, nodes, (loops[-1],))


def test_copied_foreign_or_sibling_proof_cannot_authorize_a_loop(tmp_path):
    path, analysis, nodes, loops = example(tmp_path)
    proof = analysis.worksharing_completion(PROCEDURE, nodes, (loops[0],))
    routine = analysis.routines[PROCEDURE]
    for token in (proof.public(), replace(proof)):
        with pytest.raises(CompilationError, match='(participation proof|original-team authority)'):
            extract_region(analysis, routine, loops[0], worksharing=token)
    with pytest.raises(CompilationError, match='original-team authority'):
        extract_region(analysis, routine, loops[1], worksharing=proof)
    independent = SourceEffects([path])
    other = tuple(walk(independent.routines[PROCEDURE].execution, F.Block_Nonlabel_Do_Construct))[0]
    with pytest.raises(CompilationError, match='original-team authority'):
        proof.validate(independent, PROCEDURE, (other,))
    with pytest.raises(CompilationError, match='proof token'):
        analysis.native_sections_for_nodes(PROCEDURE, (loops[0],), completion=proof)


def test_changed_source_invalidates_original_team_proof(tmp_path):
    path, analysis, nodes, loops = example(tmp_path)
    proof = analysis.worksharing_completion(PROCEDURE, nodes, (loops[0],))
    path.write_text(path.read_text().replace('end do', 'end do nowait', 1))
    with pytest.raises(CompilationError, match='changed'):
        proof.validate(analysis, PROCEDURE, (loops[0],))


def test_private_input_is_not_a_uniform_coordinate_or_scalar_capture(tmp_path):
    _, analysis, nodes, loops = example(tmp_path, BODY.replace('2*b(1,i)', 'j*b(1,i)'),
                                       opening='parallel private(i,j)')
    proof = analysis.worksharing_completion(PROCEDURE, nodes, (loops[0],))
    with pytest.raises(CompilationError, match='thread-private input'):
        extract_region(analysis, analysis.routines[PROCEDURE], loops[0], worksharing=proof)


def test_omitted_suffix_cannot_hide_per_thread_state_needed_by_later_work(tmp_path):
    body = BODY.replace('a(-2,i)=2*b(1,i)', 'j=i\na(-2,i)=2*b(1,i)').replace(
        'a(-2,i)=a(-2,i)+b(2,i)', 'a(-2,i)=a(-2,i)+j*b(2,i)')
    _, analysis, nodes, loops = example(tmp_path, body, opening='parallel private(i,j)')
    proof = analysis.worksharing_completion(PROCEDURE, nodes, (loops[0],))
    with pytest.raises(CompilationError, match='loop-written scalar is live after'):
        extract_region(analysis, analysis.routines[PROCEDURE], loops[0], worksharing=proof)


def test_one_requested_unit_does_not_revalidate_source_for_every_sibling(tmp_path, monkeypatch):
    counts = []
    for count in (2, 24):
        directory = tmp_path / str(count)
        directory.mkdir()
        body = '!$omp do\ndo i=1,n\na(-2,i)=b(1,i)\nenddo\n!$omp end do\n'
        _, analysis, nodes, loops = example(directory, body * count)
        calls = []
        original = analysis._selected_source
        def selected_source(procedure, selected):
            calls.append(1)
            return original(procedure, selected)
        monkeypatch.setattr(analysis, '_selected_source', selected_source)
        analysis.worksharing_completion(PROCEDURE, nodes, (loops[-1],))
        counts.append(len(calls))
    assert counts[0] == counts[1]
