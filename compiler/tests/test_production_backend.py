"""Production compilation cannot enter retired per-call memory/placement engines."""
import pytest
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
from compiler.frontend import lower_source
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


@pytest.mark.parametrize('policy', ['always', 'chunked', 'hybrid'])
def test_retired_public_policy_objects_are_rejected(policy):
    with pytest.raises(CompilationError, match='removed'):
        CompilerOptions(gpu_policy=policy)
    with pytest.raises(ValueError, match='removed'):
        OffloadConfig(policy=policy)


def test_retired_public_ownership_is_rejected():
    with pytest.raises(CompilationError, match='removed'):
        CompilerOptions(memory_model='call')
    with pytest.raises(CompilationError, match='removed'):
        CompilerOptions(scope_transfers='pipelined')
    with pytest.raises(ValueError, match='removed'):
        OffloadConfig(scope_transfers='pipelined')


def test_default_backend_emits_all_passes_without_ordinary_memory_or_window_planning(monkeypatch):
    import compiler.memory
    import compiler.offload.analysis
    import compiler.emission.cuda.offload

    def obsolete(*args, **kwargs):
        raise AssertionError('Production generation invoked a retired ordinary engine')

    monkeypatch.setattr(compiler.memory, 'plan_memory', obsolete)
    monkeypatch.setattr(compiler.offload.analysis, 'analyze_offload', obsolete)
    monkeypatch.setattr(compiler.emission.cuda.offload, 'generate_offload', obsolete)
    passes = ''.join(f'do i=1,n\na(i)=a(i)+{k}.d0\nenddo\n' for k in range(12))
    source = 'module many_passes\ncontains\nsubroutine advance(a,n)\nreal(8),intent(inout)::a(:)\ninteger,intent(in)::n\ninteger::i\n' + passes + 'end subroutine\nend module\n'
    function, plan = prepare_function(lower_source(source, 'advance', source_name='many_passes.f90'),
                                     options=CompilerOptions(opt_level=0))
    result = generate_sources(function, plan)
    assert len(plan.regions) == len(result.scoped['region_execution']) == 12
    assert all(r['gpu_available'] for r in result.scoped['region_execution'])
    assert result.scoped['batch_execution']['limits']['adjacent_units'] is None
    assert result.scoped['batch_execution']['candidates'] == []
    assert result.cuda is result.cpp is result.fortran is result.offload is None
    assert {'shared_entry.cu', 'shared_interface.f90', 'scoped_runtime.cu'} <= set(result.artifacts)
    with pytest.raises(CompilationError, match='removed'):
        generate_sources(function, plan, memory_model='call')
