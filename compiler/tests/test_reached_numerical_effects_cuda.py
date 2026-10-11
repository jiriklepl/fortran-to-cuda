"""Original prepared producer/native face/consumer chains agree bit for bit."""

import pytest

from compiler.tests.support.numerical_cuda import build_reached_numerical_binaries, check_prepared_chain


@pytest.fixture(scope="module", params=((4, False), (4, True), (8, False), (8, True)))
def reached_numerical_binaries(tmp_path_factory, request):
    precision, resource_fallback = request.param
    return build_reached_numerical_binaries(tmp_path_factory, precision, resource_fallback)


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "negative", "overflow", "empty", "late"])
def test_prepared_chain_keeps_native_faces_coherent_without_replay(reached_numerical_binaries, disabled, path):
    check_prepared_chain(reached_numerical_binaries, disabled, path)
