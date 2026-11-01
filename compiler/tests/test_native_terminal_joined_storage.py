"""Native completion and storage-team authority remain independent issuers."""
import pytest

from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_native_terminal_effects import issue, setup


@pytest.mark.parametrize('seal', [False, True])
def test_exact_original_terminal_team_has_independent_complete_storage_closure(tmp_path, seal):
    data = setup(tmp_path, seal=seal)
    _path, analysis, selected, _demanded = data
    reached = issue(data)
    assert not reached.barriers
    completion, = [proof for _name, _ids, proof in reached._completion
                   if type(proof).__name__ == 'NativeGraphCompletionProof']
    assert completion.public()['completion_scope'] == 'normal return only'
    team = prove_descriptor_team(analysis, 'work::owner', selected)
    assert team.selected_node_ids == completion.selected_node_ids
    storage = prove_storage_mutation_effects(analysis, 'work::owner',
                                            selection=team.selected_node_ids, joined_storage=team)
    assert storage.preserves_descriptors()
    record = storage.public()
    assert record['complete']
    assert not record['frontiers']
    assert not record['mutations']
    assert not record['escapes']
    terminal, = record['original_terminating_calls']
    assert terminal['identity'] == reached.native_graph.terminal_calls[0][1].proof.identity
    assert [row['kind'] for row in terminal['original_observable_events']] == [
        'original_external_diagnostic_write', 'original_termination']
    assert record['joined_storage_authority'] == team.public()
    assert storage.validate(joined_storage=team) is storage
    analysis.verify_publication()


def test_effect_graph_completion_cannot_replace_descriptor_team_authority(tmp_path):
    data = setup(tmp_path)
    reached = issue(data)
    completion, = [proof for _name, _ids, proof in reached._completion
                   if type(proof).__name__ == 'NativeGraphCompletionProof']
    with pytest.raises(CompilationError, match='selected team token'):
        prove_storage_mutation_effects(data[1], 'work::owner', selection=data[2], joined_storage=completion)
