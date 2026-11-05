"""Fixed original predicate points use reached host coherence, never GPU fields."""
import shutil
import subprocess
from copy import copy

import pytest

from compiler.frontend.atomic_worksharing_team import prove_atomic_worksharing_team
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.call_registration import OriginalActual, register_original_actuals
from compiler.tests.test_atomic_joined_team_source import SOURCE, emit
from compiler.tests.test_source_scopes import FACT


def point_source():
    return (SOURCE.replace('step(a,b,p,n,mode,flag)', 'step(a,b,p,n,mode,flag,gate)')
            .replace('integer::i', 'integer::i\ninteger,intent(inout)::gate(-2:2)\ninteger,parameter::index=-1')
            .replace('mode==4', 'gate(index)==4').replace('if(flag)then', 'if(gate(2)>0)then'))


def issue(tmp_path, source=None):
    path = tmp_path / 'owner.f90'
    path.write_text(source or point_source())
    analysis = SourceEffects([path])
    routine = analysis.routines['atomic_owner::step']
    nodes, = [item.original_nodes for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    return prove_atomic_worksharing_team(analysis, routine, nodes)


def test_original_parameter_and_negative_bounds_reuse_parent_point_authority(tmp_path):
    proof = issue(tmp_path)
    record = proof.public()
    assert record['version'] == 2
    points = [predicate['host_point_reads'] for predicate in record['predicates']]
    assert [point[0]['subscript'] for point in points] == [-1, 2]
    assert points[0][0]['parameter_resource'] == 'atomic_owner::step::index'
    sections = proof.predicate_sections(0)
    resource, = sections.resources
    assert resource.resource == 'argument::gate'
    assert resource.lower_bounds == (-2,)
    assert not resource.writes
    assert not resource.overwrites
    assert resource.reads[0].axes[0].lower.value == -1


@pytest.mark.parametrize('replacement', [
    ('gate(index)', 'gate(mode)'), ('gate(index)', 'gate(3)'),
    ('integer,intent(inout)::gate(-2:2)', 'real(8),intent(inout)::gate(-2:2)'),
    ('integer,intent(inout)::gate(-2:2)', 'integer,target,intent(inout)::gate(-2:2)'),
    ('integer,intent(inout)::gate(-2:2)', 'integer,intent(inout)::gate(:)'),
    ('a(i)=a(i)+p(i-1)', 'gate(1)=1\na(i)=a(i)+p(i-1)'),
])
def test_unproved_or_conflicting_predicate_storage_rejects(tmp_path, replacement):
    with pytest.raises(CompilationError):
        issue(tmp_path, point_source().replace(*replacement))


@pytest.mark.parametrize('fault', ['copy', 'bool_index', 'negative_index', 'public_edit', 'registry'])
def test_point_accessor_retains_exact_original_proof_authority(tmp_path, fault):
    proof = issue(tmp_path)
    if fault == 'public_edit':
        proof.public()['predicates'][0]['host_point_reads'][0]['subscript'] = 99
        assert proof.predicate_sections(0).resources[0].reads[0].axes[0].lower.value == -1
        return
    if fault == 'copy':
        proof = copy(proof)
    elif fault == 'registry':
        proof._analysis._atomic_worksharing_teams.pop(proof.identity)
    with pytest.raises(CompilationError):
        proof.predicate_sections(True if fault == 'bool_index' else -1 if fault == 'negative_index' else 0)


def test_reached_host_transactions_precede_original_predicate_and_exclude_gpu_abi(tmp_path):
    outputs, report, text = emit(tmp_path, source=point_source(), extra_captures={'argument::gate': dict(FACT)})
    scope, = report['scopes']
    task, = scope['source_device_tasks']
    assert task['version'] == 2
    assert 'argument::gate' in task['execution_binding']['canonical_resources']
    assert all('argument::gate' not in [field['resource'] for field in variant['provider']['metadata']['fields']]
               for variant in task['alternatives'])
    first = text.index('fort_atomic_predicate_0 = gate(index) == 4')
    assert text.index('fort_scope_access_begin_v1', text.index('fort_atomic_select: block')) < first
    fatal = text.index("error stop 'atomic predicate acquisition failed fatally'")
    assert fatal < text.index('if (fort_status /= FORT_SCOPE_OK) exit fort_atomic_select', fatal) < first
    admission = text.index('fort_scope_admit_source_reads_v1', text.index('fort_atomic_select: block'))
    assert admission < text.index('fort_scope_access_begin_v1', admission) < first
    fatal_read = text.index("error stop 'atomic predicate source admission failed fatally'")
    assert admission < fatal_read < text.index('exit fort_atomic_select', fatal_read) < first
    assert first < text.index('fort_scope_host_end', first) < text.index('if (fort_atomic_predicate_0) then')
    assert text.index('if (fort_atomic_predicate_0) then') < text.index('fort_atomic_predicate_1 = gate(2) > 0')
    assert text.index('fort_atomic_predicate_1 = gate(2) > 0') < text.index('select case(fort_atomic_variant)')
    assert text.count('!$omp parallel private(i)') == 1
    assert 'atomic original team attempted work; replay prohibited' in text
    fc = shutil.which('gfortran-15')
    assert fc is not None
    generated = tmp_path / 'generated'
    for name, content in outputs.items():
        path = generated / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    roles = {name: index for index, name in enumerate(('common_runtime', 'shared_entry', 'original_source'))}
    for index, artifact in sorted(enumerate(report['build_sources']), key=lambda pair: roles[pair[1]['role']]):
        if artifact['language'] != 'fortran':
            continue
        result = subprocess.run([fc, '-fopenmp', '-ffree-line-length-none', '-J', str(generated), '-I', str(generated),
            '-c', str(generated / artifact['path']), '-o', str(generated / f'host-{index}.o')],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


def test_same_patch_count_registration_placeholders_render_own_all_reservations(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import compiler.scopes.provenance
    import compiler.scopes.segments
    from compiler.scopes.call_registration import finish_original_registrations

    proof = issue(tmp_path, point_source().replace('integer::i\n', 'integer::i\ninteger::other(-4:4)\n')
                  .replace('if(gate(2)>0)then', 'if(other(0)>0)then'))
    routine = proof._routine
    bindings = routine.scope.bindings
    # This isolates reservation transport. Original bindings/references remain
    # authentic; eligibility and trace rendering are outside the key regression.
    monkeypatch.setattr(compiler.scopes.segments, '_host_read_storage_candidate', lambda *_: True)
    monkeypatch.setattr(compiler.scopes.provenance, 'position', lambda *_args, **_kwargs: [])
    builder = SimpleNamespace(config=SimpleNamespace(scope_execution='graph'), device_budget=4096,
        entry=routine, capture_many=lambda values: [dict(FACT) for _ in values],
        owner_transfer_setup=lambda *_args, **_kwargs: [], provider_only_lifetime=None)
    class Owner:
        patches = []
        context = 'fort_context'
        enabled = 'fort_enabled'
        epoch_pristine = 'fort_epoch_pristine'
        leaves = []
        def __init__(self):
            self.owner = self
            self.resources = {}
            self.scalar_element_resources = {}
        def declaration(self, _binding):
            pass
        def add_resource(self, binding):
            self.resources.setdefault(binding.root, (binding, len(self.resources) + 1))
        def registration_identity(self, number):
            return str(number) + '_c_int64_t'
        def creation_trace_owner(self):
            return 1
        def close(self, **_kwargs):
            return []
    owner = Owner()
    owner.routine, owner.builder = routine, builder
    refs = [proof.predicate_sections(index).resources[0].reads[0].node for index in (0, 1)]
    def native(name):
        return SimpleNamespace(binding=bindings[name], actual=name,
                                             formal_binding=SimpleNamespace(intent='in'))
    def element(name, actual, bounds):
        return SimpleNamespace(binding=bindings[name], actual=actual, bounds=bounds)
    first = register_original_actuals(owner, (OriginalActual(bindings['a'], 'a'),),
        native_arrays=(native('p'),), scalar_elements=(element('gate', refs[0], (-2, 2)),))
    second = register_original_actuals(owner, (OriginalActual(bindings['b'], 'b'),),
        native_arrays=(native('a'),), scalar_elements=(element('other', refs[1], (-4, 4)),))
    assert first != second
    assert len(owner.patches) == 0
    rendered = finish_original_registrations(owner, first[0] + '\n!SECOND\n' + second[0])
    a, b = rendered.split('!SECOND')
    assert 'shape(a,' in a
    assert 'shape(p,' in a
    assert 'c_loc(gate)' in a
    assert 'shape(b,' not in a
    assert 'c_loc(other)' not in a
    assert 'shape(b,' in b
    assert 'shape(a,' in b
    assert 'c_loc(other)' in b
    assert 'shape(p,' not in b
    assert 'c_loc(gate)' not in b
    assert first[0] not in rendered
    assert second[0] not in rendered
    import compiler.scopes.call_registration as registration
    monkeypatch.setattr(registration, '_name', lambda *_args: first[0][1:])
    with pytest.raises(CompilationError, match='identity collision'):
        register_original_actuals(owner, ())


def test_dormant_predicate_registration_stays_inside_original_scalar_branch(tmp_path):
    from compiler.tests.test_atomic_worksharing_team import BODY, unit
    body = 'if(flag)then\n' + BODY + 'else\n' + unit('a', 'a(i)+p(i)') + 'endif\n'
    source = point_source().replace(BODY.replace('mode==4', 'gate(index)==4')
                                   .replace('if(flag)then', 'if(gate(2)>0)then'),
                                   body.replace('mode==4', 'gate(index)==4'))
    _outputs, report, text = emit(tmp_path, source=source,
        extra_captures={'argument::gate': {**FACT, 'initialized': 'none', 'lifetime': 'invocation'}})
    scope, = report['scopes']
    task, = scope['source_device_tasks']
    assert task['source_proof']['predicates'][0]['host_point_reads'] == []
    selected = text.split('fort_atomic_select: block', 1)[1].split('select case(fort_atomic_variant)', 1)[0]
    assert selected.index('IF (flag) THEN') < selected.index('shape(gate,')
    assert selected.index('shape(gate,') < selected.index('fort_atomic_predicate_1 = gate(index) == 4')


def test_reached_none_entry_predicate_has_exact_read_admission_before_host_acquisition(tmp_path):
    _outputs, report, text = emit(tmp_path, source=point_source(),
        extra_captures={'argument::gate': {**FACT, 'initialized': 'none', 'lifetime': 'invocation'}})
    task, = report['scopes'][0]['source_device_tasks']
    assert 'only missing exact read regions' in task['source_proof']['preparation']
    assert 'no writes or whole-entry definitions' in task['source_proof']['preparation']
    selected = text.split('fort_atomic_select: block', 1)[1].split('select case(fort_atomic_variant)', 1)[0]
    begin = -1
    for index in (0, 1):
        # One exact rank-one point access is reused for admission and acquire.
        begin = selected.index('fort_scope_admit_source_reads_v1', begin + 1)
        assert begin < selected.index('fort_scope_access_begin_v1', begin)
        assert selected.index('fort_scope_access_begin_v1', begin) < selected.index(f'fort_atomic_predicate_{index} =')
    assert selected.count('fort_scope_admit_source_reads_v1') == 2
    assert selected.index('if (fort_atomic_predicate_0) then') < selected.rindex('fort_scope_admit_source_reads_v1')
