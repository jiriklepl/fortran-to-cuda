"""Whole original joins have a source budget, without numerical-unit credit."""
from copy import copy
from types import MappingProxyType

import pytest

from compiler.frontend import resource_reachability as rr
from compiler.frontend.native_completion import NativeCompletionProof
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(directory, statements, *, joined=True, sealed=False, operations=256):
    directory.mkdir(parents=True, exist_ok=True)
    if joined:
        assert statements >= 6
        writes = statements - 4
        counts = (writes // 2, writes - writes // 2)
        body = '!$omp parallel private(i)\n'
        for count in counts:
            body += '!$omp do schedule(runtime)\ndo i=1,n\n'
            body += 'a(i)=a(i)+0.125_8\n' * count
            body += 'enddo\n!$omp end do\n'
        body += '!$omp end parallel\n'
    else:
        body = 'a(1)=a(1)+0.125_8\n' * statements
    source = ('module budget_source\nimplicit none\ncontains\nsubroutine step(a,n)\n'
              'real(8),intent(inout)::a(:)\ninteger,intent(in)::n\ninteger::i\n' + body
              + 'end subroutine\nend module\n')
    path = directory / 'source.f90'
    path.write_text(source)
    analysis = SourceEffects([path], operations=operations)
    if sealed:
        analysis.seal_session()
    routine = analysis.routines['budget_source::step']
    if joined:
        group, = [node for node in grouped_original_sequence(routine.execution.content, {})
                  if isinstance(node, NativeGroupCandidate)]
        selected = group.original_nodes
    else:
        selected = tuple(routine.execution.content)
    assert analysis.operation_limit == operations
    return path, analysis, routine, selected


def issue(parts, *, project_payload=True):
    _path, analysis, routine, selected = parts
    return rr.prove_native_effect_graph(analysis, routine.qualified, selected, project_payload=project_payload)


@pytest.mark.parametrize('sealed', [False, True])
@pytest.mark.parametrize('statements', [256, 257, 486])
def test_exact_whole_multido_group_retains_every_statement_beyond_kernel_window(tmp_path, statements, sealed):
    parts = fixture(tmp_path, statements, sealed=sealed)
    graph = issue(parts)
    _path, analysis, routine, selected = parts
    counts = dict(graph.source_work)
    assert counts['statements'] == statements
    assert analysis.operation_limit == 256
    template, = graph.templates
    source = dict(template.source_work)
    assert source['ordinary_statements'] == 0
    assert source['joined_source_statements'] == statements
    assert source['syntax_nodes'] < rr.NATIVE_GROUP_SCAN_LIMIT == 65536
    receipt, = template._source_budget_receipts
    assert receipt[2] is template.completion[0][2]
    assert type(receipt[2]) is NativeCompletionProof
    assert graph.validate(analysis, routine.qualified, selected) is graph
    assert issue(parts) is graph
    public = graph.public()
    assert public['limits']['per_template']['ordinary_statements'] == 256
    assert public['limits']['per_template']['source_statements'] == 65536
    assert public['limits']['joined_source_budget_policy_version'] == rr.JOINED_SOURCE_BUDGET_VERSION
    assert all(public[name] is False for name in ('execution_authority', 'alias_authority',
                                                  'internal_cut_authority', 'motion_authority'))
    nodes = list(template.nodes)
    assignments = []
    while nodes:
        node = nodes.pop()
        if isinstance(node, rr.NativeEffectNode):
            nodes.extend(node.items)
            if node.kind == 'Assignment_Stmt':
                assignments.append(node.original)
    assert len(assignments) == len(set(map(id, assignments))) == statements - 4


@pytest.mark.parametrize('statements', [256, 257])
def test_ordinary_non_group_source_keeps_configured_kernel_operation_limit(tmp_path, statements):
    parts = fixture(tmp_path, statements, joined=False)
    if statements == 257:
        scan = rr._Scan(parts[1], ())
        with pytest.raises(rr.ResourceReachabilityBudgetExceeded, match='source scan budget'):
            scan.routine(parts[2].qualified, {}, parts[3])
    else:
        graph = issue(parts)
        assert dict(graph.source_work)['statements'] == 256
        assert not graph.templates[0]._source_budget_receipts


def test_control_only_discovery_never_issues_the_ordinary_join_for_its_budget(tmp_path, monkeypatch):
    parts = fixture(tmp_path, 257)
    def forbidden(*args):
        pytest.fail('control-only graph invoked ordinary payload completion')
    monkeypatch.setattr(parts[1], 'joined_completion', forbidden)
    with pytest.raises(rr.ResourceReachabilityBudgetExceeded, match='source scan budget'):
        issue(parts, project_payload=False)


@pytest.mark.parametrize('receipt_change', ['absent', 'foreign', 'copy', 'selection', 'record'])
def test_unrelated_or_changed_ordinary_receipt_cannot_extend_source_budget(tmp_path, monkeypatch, receipt_change):
    parts = fixture(tmp_path / 'owner', 257)
    _path, analysis, routine, selected = parts
    ordinary = analysis.joined_completion(routine.qualified, selected)
    if receipt_change == 'absent':
        def unavailable(*args):
            raise CompilationError('no ordinary join')
        monkeypatch.setattr(analysis, 'joined_completion', unavailable)
    else:
        if receipt_change == 'foreign':
            other = fixture(tmp_path / 'other', 257)
            ordinary = other[1].joined_completion(other[2].qualified, other[3])
        elif receipt_change == 'copy':
            ordinary = copy(ordinary)
        elif receipt_change == 'selection':
            object.__setattr__(ordinary, 'selected_node_ids', ordinary.selected_node_ids[:-1])
        elif receipt_change == 'record':
            object.__setattr__(ordinary, '_record', MappingProxyType({}))
        monkeypatch.setattr(analysis, 'joined_completion', lambda *args: ordinary)
    with pytest.raises(rr.ResourceReachabilityBudgetExceeded, match='source scan budget'):
        issue(parts)


@pytest.mark.parametrize('selection_change', ['missing_join', 'copied_header'])
def test_partial_or_foreign_parser_group_has_no_original_joined_budget(tmp_path, selection_change):
    parts = fixture(tmp_path, 257)
    path, analysis, routine, selected = parts
    if selection_change == 'missing_join':
        altered = selected[:-1]
    else:
        other = fixture(tmp_path / 'other', 257)
        assert other[3][0] is not selected[0]
        altered = (other[3][0], *selected[1:])
    with pytest.raises(CompilationError):
        issue((path, analysis, routine, altered))


@pytest.mark.parametrize('mutation', ['registry_entry', 'registry_container', 'proof', 'group', 'receipt_attachment'])
def test_cached_template_revalidates_its_group_budget_before_reuse(tmp_path, mutation):
    parts = fixture(tmp_path, 486)
    graph = issue(parts)
    path, analysis, routine, selected = parts
    template, = graph.templates
    receipt, = template._source_budget_receipts
    ordinary = receipt[2]
    if mutation == 'registry_entry':
        analysis._joined_completions[ordinary.identity] = copy(ordinary)
    elif mutation == 'registry_container':
        analysis._joined_completions = dict(analysis._joined_completions)
    elif mutation == 'proof':
        object.__setattr__(ordinary, '_record', MappingProxyType({}))
    elif mutation == 'group':
        path.write_text(path.read_text() + '! changed captured input\n')
    else:
        object.__setattr__(template, '_source_budget_receipts', ())
    with pytest.raises(CompilationError):
        graph.validate(analysis, routine.qualified, selected)
    with pytest.raises(CompilationError):
        issue(parts)


def test_changed_source_budget_policy_invalidates_cached_templates_and_graphs(tmp_path, monkeypatch):
    parts = fixture(tmp_path, 486)
    first = issue(parts)
    _path, analysis, routine, selected = parts
    monkeypatch.setattr(rr, 'JOINED_SOURCE_BUDGET_VERSION', rr.JOINED_SOURCE_BUDGET_VERSION + 1)
    with pytest.raises(CompilationError):
        first.validate(analysis, routine.qualified, selected)
    second = issue(parts)
    assert second is not first
    assert second.identity != first.identity
    assert second.templates[0] is not first.templates[0]
    assert dict(second.source_work)['statements'] == 486


def test_changed_operation_policy_invalidates_cached_group_without_rewriting_configuration(tmp_path):
    parts = fixture(tmp_path, 486)
    first = issue(parts)
    _path, analysis, routine, selected = parts
    analysis.operation_limit = 255
    with pytest.raises(CompilationError):
        first.validate(analysis, routine.qualified, selected)
    with pytest.raises(CompilationError):
        issue(parts)
    fresh = fixture(tmp_path / 'fresh', 486, operations=255)
    second = issue(fresh)
    assert second is not first
    assert dict(second.source_work)['statements'] == 486
    assert fresh[1].operation_limit == analysis.operation_limit == 255


def test_authentic_join_does_not_bypass_universal_syntax_limit(tmp_path, monkeypatch):
    parts = fixture(tmp_path, 486)
    monkeypatch.setattr(rr, 'NATIVE_GROUP_SCAN_LIMIT', 1000)
    with pytest.raises(rr.ResourceReachabilityBudgetExceeded, match='source scan budget'):
        issue(parts)


@pytest.mark.parametrize('mutation', ['registry', 'record', 'selection','policy','operations'])
def test_final_receipt_fence_rejects_mutation_after_delegated_publication(tmp_path, monkeypatch, mutation):
    parts = fixture(tmp_path, 486)
    graph = issue(parts)
    _path, analysis, routine, selected = parts
    receipt, = graph.templates[0]._source_budget_receipts
    ordinary = receipt[2]
    original = NativeCompletionProof.public
    changed = []

    def late(proof):
        result = original(proof)
        if proof is ordinary and not changed:
            if mutation == 'registry':
                del analysis._joined_completions[proof.identity]
            elif mutation == 'record':
                object.__setattr__(proof, '_record', MappingProxyType({}))
            elif mutation=='selection':
                object.__setattr__(proof, 'selected_node_ids', ())
            elif mutation=='policy':
                monkeypatch.setattr(rr,'JOINED_SOURCE_BUDGET_VERSION',rr.JOINED_SOURCE_BUDGET_VERSION+1)
            else:
                analysis.operation_limit-=1
            changed.append(True)
        return result

    monkeypatch.setattr(NativeCompletionProof, 'public', late)
    with pytest.raises(CompilationError):
        graph.validate(analysis, routine.qualified, selected)
    assert changed
