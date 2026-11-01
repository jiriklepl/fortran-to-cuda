"""Proof metadata borrows only deeply immutable session input provenance."""
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.frontend.structured_effects import _freeze_source_record
from compiler.ir import CompilationError


def fixture(tmp_path, *, sealed=True):
    path = tmp_path / "original.f90"
    prepared = tmp_path / "prepared.f90"
    source = """module provenance
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
a(:)=a(:)+1.0_8
end subroutine
end module
""" + "! configured source line\n" * 4096
    path.write_text(source)
    prepared.write_text(source)
    digest = sha256(path.read_bytes()).hexdigest()
    document = {"schema_version": 1, "source_inputs": {str(path): digest}, "preserves_source_order": True,
                "dependencies": {}, "configuration": {"compiler": "original configured test", "options": ["fixed"]},
                "entries": [{"source": str(path), "path": str(prepared), "sha256": digest,
                             "line_map": list(range(1, len(source.splitlines()) + 1))}]}
    analysis = SourceEffects([path], analysis_sources=document)
    if sealed:
        analysis.seal_session()
    return path, prepared, analysis


def test_session_proof_shares_deep_immutable_provenance_but_public_metadata_is_detached(tmp_path):
    path, _prepared, analysis = fixture(tmp_path)
    token = prove_storage_mutation_effects(analysis, "provenance::advance")
    provenance = token._record["source_inputs"]
    assert provenance["sources"] is analysis.inputs.sources
    assert provenance["configuration"] is analysis.inputs.configuration
    assert provenance["entries"][0] is analysis.inputs.entries[str(path)]
    with pytest.raises(CompilationError, match="immutable"):
        provenance["entries"][0]["line_map"][0] = 4
    public = token.public()
    public["source_inputs"]["entries"][0]["line_map"][0] = 4
    public["source_inputs"]["configuration"]["options"].append("different")
    public["resources"].clear()
    assert token.validate().preserves_descriptors()
    assert token.public()["source_inputs"]["entries"][0]["line_map"][0] == 1
    assert token.public()["resources"] == ["argument::a"]


def test_standalone_proof_provenance_is_fully_detached_and_metadata_changes_reject(tmp_path):
    _path, _prepared, analysis = fixture(tmp_path, sealed=False)
    token = prove_storage_mutation_effects(analysis, "provenance::advance")
    assert token._record["source_inputs"]["sources"] is not analysis.inputs.sources
    original = token.public()
    analysis.inputs.configuration["options"].append("changed")
    assert token.public() == original
    with pytest.raises(CompilationError):
        token.validate()


@pytest.mark.parametrize("input_path", ["original", "prepared"])
def test_shared_provenance_does_not_replace_the_live_publication_fence(tmp_path, input_path):
    path, prepared, analysis = fixture(tmp_path)
    token = prove_storage_mutation_effects(analysis, "provenance::advance")
    original_facts = token.public()
    changed = path if input_path == "original" else prepared
    changed.write_text(changed.read_text() + "! changed after issuance\n")
    assert token.validate() is token
    assert token.public() == original_facts
    with pytest.raises(CompilationError, match="source changed"):
        analysis.verify_publication()


def test_equal_looking_foreign_input_registry_and_changed_provenance_do_not_borrow_session(tmp_path):
    _path, _prepared, analysis = fixture(tmp_path)
    record = {"source_inputs": analysis.inputs.identity(), "private": {"values": [1, 2]}}
    frozen = _freeze_source_record(analysis, record)
    record["private"]["values"].append(3)
    assert frozen["private"]["values"] == (1, 2)
    changed = deepcopy(analysis.inputs.identity())
    changed["configuration"]["options"].append("changed")
    with pytest.raises(CompilationError, match="provenance differs"):
        _freeze_source_record(analysis, {"source_inputs": changed})
    original_inputs = analysis.inputs
    replacement = deepcopy(original_inputs)
    with pytest.raises(CompilationError, match="original source registry is immutable"):
        analysis.inputs = replacement
    assert analysis.inputs is original_inputs
    assert _freeze_source_record(analysis, {"source_inputs": record["source_inputs"]})["source_inputs"] == frozen["source_inputs"]
    # Retain the consumer fence under deliberate internal root corruption.
    object.__setattr__(analysis, "inputs", replacement)
    with pytest.raises(CompilationError, match="original registry changed"):
        _freeze_source_record(analysis, {"source_inputs": record["source_inputs"]})
