"""Production compilation cannot enter retired per-call memory/placement engines."""
import pytest

from compiler.driver.options import CompilerOptions
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
        CompilerOptions(scope_transfers='pipelined')
    with pytest.raises(ValueError, match='removed'):
        OffloadConfig(scope_transfers='pipelined')
