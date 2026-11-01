"""Immutable source provenance shares storage while preserving exact JSON."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.frontend import source_inputs
from compiler.frontend.source_inputs import SourceInputs
from compiler.ir import CompilationError
from compiler.tests.test_source_session import configured


def original_identity(inputs):
    return deepcopy({"schema_version": 1, "sources": inputs.sources,
                     "dependencies": inputs.dependencies, "configuration": inputs.configuration,
                     "entries": [inputs.entries[name] for name in sorted(inputs.entries)]})


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_sealed_identity_preserves_exact_json_and_digest(tmp_path):
    analysis, *_ = configured(tmp_path)
    original = original_identity(analysis.inputs)
    current = analysis.inputs.identity()
    assert encoded(current) == encoded(original)
    assert sha256(encoded(current).encode()).digest() == sha256(encoded(original).encode()).digest()
    assert current["sources"] is analysis.inputs.sources
    assert current["dependencies"] is analysis.inputs.dependencies
    assert current["configuration"] is analysis.inputs.configuration
    assert current["entries"][0] is next(iter(analysis.inputs.entries.values()))


def test_sealed_queries_do_not_copy_line_maps(tmp_path, monkeypatch):
    analysis, *_ = configured(tmp_path)
    expected = encoded(original_identity(analysis.inputs))

    def unexpected(*args, **kwargs):
        pytest.fail("sealed source provenance was copied again")

    monkeypatch.setattr(source_inputs, "deepcopy", unexpected)
    for _ in range(3):
        assert encoded(analysis.inputs.identity()) == expected


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(schema_version=2),
    lambda value: value["sources"].clear(),
    lambda value: value["dependencies"].clear(),
    lambda value: value["configuration"]["selected"].append("other"),
    lambda value: value["entries"].clear(),
    lambda value: value["entries"][0].update(sha256="other"),
    lambda value: value["entries"][0]["line_map"].append(1),
])
def test_no_returned_container_can_mutate_sealed_authority(tmp_path, mutate):
    analysis, *_ = configured(tmp_path)
    before = encoded(analysis.inputs.identity())
    with pytest.raises(CompilationError, match="immutable"):
        mutate(analysis.inputs.identity())
    assert encoded(analysis.inputs.identity()) == before


def test_standalone_identity_and_public_exports_stay_detached_and_mutable(tmp_path):
    analysis, original, *_ = configured(tmp_path)
    exported = analysis.inputs.public()
    inputs = SourceInputs([original], exported)
    before = encoded(inputs.identity())
    private = inputs.identity()
    private["configuration"]["selected"].append("private")
    private["entries"][0]["line_map"].clear()
    private["sources"].clear()
    assert encoded(inputs.identity()) == before
    exported["configuration"]["selected"].append("other")
    assert encoded(inputs.identity()) == before
    assert list(analysis.inputs.configuration["selected"]) == ["original"]


def test_private_input_copy_is_unsealed_and_does_not_retain_authority_aliases(tmp_path):
    analysis, *_ = configured(tmp_path)
    copied = deepcopy(analysis.inputs)
    original = encoded(analysis.inputs.identity())
    copied.configuration["selected"].append("private")
    copied.entries[next(iter(copied.entries))]["line_map"].clear()
    assert encoded(copied.identity()) != original
    assert encoded(analysis.inputs.identity()) == original


def test_shared_identity_does_not_bypass_live_publication_checks(tmp_path):
    analysis, _original, prepared, *_ = configured(tmp_path)
    identity = encoded(analysis.inputs.identity())
    prepared.write_text(prepared.read_text() + "! modified live source\n")
    assert encoded(analysis.inputs.identity()) == identity
    with pytest.raises(CompilationError, match="analysis source changed"):
        analysis.verify_publication()
