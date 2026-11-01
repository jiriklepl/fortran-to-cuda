"""Every graph context gets the live planner before any recorded work."""
from types import SimpleNamespace

import pytest

from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder


def test_source_or_external_only_graph_uses_existing_runtime_setup_without_worker():
    builder = SimpleNamespace(config=OffloadConfig(policy='sections'))
    imports = []
    code = '\n'.join(ScopeBuilder.owner_transfer_setup(builder, set(), imports,
                                                     context='ctx', status='status'))
    assert "bind(C, name='fort_scope_task_graph_enable_v1')" in code
    assert 'integer(c_int64_t), value :: context' in code
    assert 'if (status == FORT_SCOPE_OK) then' in code
    assert code.count('status = fort_enable_checked_graph(ctx)') == 1
    assert imports == []  # The source-only path has no numerical worker module.


def test_ordinary_worker_configuration_remains_the_single_graph_setup():
    public = {'fortran_module': 'qualified_worker',
              'transfer_configuration': {'fortran_procedure': 'configure_worker'}}
    builder = SimpleNamespace(config=OffloadConfig(policy='sections'),
                              entry_artifacts=lambda name: (public, None))
    imports = []
    code = ScopeBuilder.owner_transfer_setup(builder, {'module::worker'}, imports)
    assert code == ['if (fort_status == FORT_SCOPE_OK) &', '  fort_status = fort_configure(fort_context)']
    assert imports == ['use qualified_worker, only: fort_configure => configure_worker']


def test_source_only_graph_does_not_invent_an_unqualified_transfer_protocol():
    builder = SimpleNamespace(config=OffloadConfig(policy='sections', scope_transfers='pinned'))
    with pytest.raises(CompilationError, match='direct transport'):
        ScopeBuilder.owner_transfer_setup(builder, set(), [])
