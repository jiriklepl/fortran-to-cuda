"""Canonical fast encoding preserves mutable-profile authentication and limits."""
import json
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend import numerical_preparation as preparation
from compiler.ir import CompilationError
from compiler.numerical_contract import numerical_build_contract
from compiler.offload.config import OffloadConfig


def streamed(profile):
    return ''.join(json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
                   .iterencode(profile)).encode('ascii')


@pytest.mark.parametrize('profile', [
    {}, {'nested': {'list': [None, True, False, 0, -1, 0.0, -0.0, 1e-300, 1e300]}},
    {'escaped': 'a\\b\n\t"\u0000', 'unicode': 'Žluťoučký雪😀'},
    {4: 'four', -2: {'x': (1, 2, 3)}},
    {'rows': [{'x': n, 'label': 'measurement'} for n in range(1000)]},
])
def test_fast_identity_matches_original_stream_and_byte_accounting(profile):
    encoded = streamed(profile)
    charged = []
    actual = preparation._profile_identity(profile, preparation._Budget(charged.append))
    assert actual == {'private_identity_kind': 'calibration_profile_sha256_v1',
                      'bytes': len(encoded), 'sha256': sha256(encoded).hexdigest()}
    assert charged == [1] * (1 + len(encoded) // 4096)


def test_none_profile_remains_absent_without_budget_charge():
    charged = []
    assert preparation._profile_identity(None, preparation._Budget(charged.append)) is None
    assert charged == []


@pytest.mark.parametrize('defect', ['nan', 'infinity', 'object', 'mixed_keys', 'cycle'])
def test_fast_encoder_rejects_the_same_noncanonical_profile_values(defect):
    profile = {'raw': {'nan': float('nan'), 'infinity': float('inf'), 'object': object(),
                       'mixed_keys': {'text': 1, 4: 2}, 'cycle': None}[defect]}
    if defect == 'cycle':
        profile['raw'] = profile
    with pytest.raises((TypeError, ValueError, OverflowError, RecursionError)):
        streamed(profile)
    with pytest.raises(CompilationError, match='invalid.*profile identity'):
        preparation._profile_identity(profile, preparation._Budget())


def test_exact_encoded_byte_limit_including_unicode_escapes(monkeypatch):
    profile = {'raw': '雪' * 1000}
    size = len(streamed(profile))
    monkeypatch.setattr(preparation, '_PROFILE_TEXT_LIMIT', size)
    assert preparation._profile_identity(profile, preparation._Budget())['bytes'] == size
    monkeypatch.setattr(preparation, '_PROFILE_TEXT_LIMIT', size - 1)
    with pytest.raises(CompilationError, match='profile text budget exhausted'):
        preparation._profile_identity(profile, preparation._Budget())


def test_callback_mutation_after_encoded_prefix_still_fails_fresh_second_digest():
    profile = {'aaa': 1, 'observations': list(range(3000)),
               'numerical_contract': numerical_build_contract()}
    config = OffloadConfig(profile=profile)
    calls = 0

    def mutate(_count):
        nonlocal calls
        calls += 1
        if calls == 2:
            profile['aaa'] = 2

    with pytest.raises(CompilationError, match='profile changed while accounting'):
        preparation._configuration(CompilerOptions(), config, 'm::p', {}, preparation._Budget(mutate))
    assert profile['aaa'] == 2
    assert calls > 2


def test_profile_budget_callback_failure_propagates_without_relaxation():
    def reject(_count):
        raise CompilationError('owned aggregate allowance exhausted')

    with pytest.raises(CompilationError, match='owned aggregate allowance exhausted'):
        preparation._profile_identity({'values': list(range(3000))}, preparation._Budget(reject))
